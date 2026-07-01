from pathlib import Path
import zipfile

import numpy as np
from scipy.io import savemat

from backend.papers import PaperCandidate, PaperMaterial, finalize_paper_workflow
from fnirs_core.data import NIRSData, NIRSDataError, load_fNIRS_data
from fnirs_core.data import make_demo_nirs_data
from fnirs_core.experiments import ExperimentConfig, run_experiment
from fnirs_core.knowledge import KnowledgeBase, KnowledgeBaseError, extract_text_from_document
from fnirs_core.preprocessing import PreprocessingPipeline


def test_knowledge_base_fallback_search(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "guide.md").write_text(
        "# fNIRS\n\nfNIRS preprocessing uses TDDR, Beer-Lambert conversion, and subject-wise LOSO validation.",
        encoding="utf-8",
    )
    kb = KnowledgeBase([knowledge_dir], vector_store_dir=tmp_path / "vectors")
    kb.refresh()
    results = kb.search("fNIRS LOSO preprocessing", top_k=2)
    assert results
    assert "LOSO" in results[0].content


def test_knowledge_chunks_preserve_overlap(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    long_text = "abcdefghijklmnopqrstuvwxyz" * 12
    (knowledge_dir / "guide.md").write_text(long_text, encoding="utf-8")
    kb = KnowledgeBase([knowledge_dir], vector_store_dir=tmp_path / "vectors", chunk_size=200, chunk_overlap=25)
    kb.refresh()
    chunks = kb.get_document_chunks(kb.list_documents()[0].id)
    assert chunks[0].content[-25:] == chunks[1].content[:25]


def test_knowledge_index_invalidates_when_embedding_model_changes(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "guide.md").write_text("fNIRS LOSO validation", encoding="utf-8")
    vector_dir = tmp_path / "vectors"

    kb = KnowledgeBase([knowledge_dir], vector_store_dir=vector_dir, embedding_model="embed-a")
    kb.refresh()
    metadata = kb._load_metadata()
    assert metadata["configured_embedding_model"] == "embed-a"

    changed = KnowledgeBase([knowledge_dir], vector_store_dir=vector_dir, embedding_model="embed-b")
    assert not changed._metadata_matches_sources(metadata)


def test_knowledge_zip_rejects_unsafe_paths(tmp_path: Path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.md", "unsafe")
    try:
        extract_text_from_document(archive_path)
    except KnowledgeBaseError as exc:
        assert "unsafe path" in str(exc)
    else:
        raise AssertionError("unsafe zip path should be rejected")


def test_extract_text_decodes_gb18030_chinese_document(tmp_path: Path):
    text_path = tmp_path / "中文知识.txt"
    original = "近红外脑功能成像知识库：血氧信号、通道质量、运动伪影校正。"
    text_path.write_bytes(original.encode("gb18030"))

    extracted = extract_text_from_document(text_path)

    assert "近红外脑功能成像知识库" in extracted
    assert "运动伪影校正" in extracted
    assert "乱码" not in extracted


def test_pdf_glyph_name_extraction_is_rejected():
    pdf_path = Path("artifacts/knowledge_uploads/raw/create_pdf.aspx.pdf")
    if not pdf_path.exists():
        return

    try:
        extract_text_from_document(pdf_path)
    except KnowledgeBaseError as exc:
        assert "font glyph codes" in str(exc)
    else:
        raise AssertionError("PDF glyph codes should not be accepted as extracted text")


def test_knowledge_refresh_skips_existing_pdf_glyph_name_markdown(tmp_path: Path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "good.md").write_text("fNIRS 中文知识库包含血氧信号和运动伪影校正。", encoding="utf-8")
    (knowledge_dir / "bad.md").write_text(
        "/G34/G35/GC8/GC9/GCA/G9E/GCB/GCC/G46/GCF/G97/G28/G29/G2A/G4C/G56/GAE/GAF",
        encoding="utf-8",
    )

    kb = KnowledgeBase([knowledge_dir], vector_store_dir=tmp_path / "vectors")
    kb.refresh()

    assert kb.list_sources() == ["good.md"]


def test_add_or_update_document_embeds_only_changed_document(tmp_path: Path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "old.md").write_text("old fNIRS LOSO preprocessing note", encoding="utf-8")
    vector_dir = tmp_path / "vectors"
    kb = KnowledgeBase([knowledge_dir], vector_store_dir=vector_dir, chunk_size=200)
    kb.refresh()

    calls = []
    original_embed = KnowledgeBase._embed_texts

    def recording_embed(self, texts):
        calls.append(list(texts))
        return original_embed(self, texts)

    monkeypatch.setattr(KnowledgeBase, "_embed_texts", recording_embed)
    (knowledge_dir / "new.md").write_text("new 中文知识包含血氧信号和运动伪影校正。", encoding="utf-8")

    kb = KnowledgeBase([knowledge_dir], vector_store_dir=vector_dir, chunk_size=200)
    kb.add_or_update_document(knowledge_dir / "new.md")

    assert len(calls) == 1
    assert len(calls[0]) == 1
    assert "new" in calls[0][0]
    assert sorted(kb.list_sources()) == ["new.md", "old.md"]


def test_preprocessing_demo_epochs():
    data = make_demo_nirs_data(n_subjects=3, trials_per_subject=4)
    result = PreprocessingPipeline({"epoch_start": 0.0, "epoch_end": 5.0}).run(data)
    assert result.epochs.ndim == 4
    assert result.summary["n_epochs"] > 0
    assert result.summary["subject_count"] == 3


def test_no_event_recording_fallback_produces_two_placeholder_labels():
    data = make_demo_nirs_data(n_subjects=1, trials_per_subject=1)
    no_event = NIRSData(
        raw_data=data.raw_data,
        sampling_rate=data.sampling_rate,
        channel_names=data.channel_names,
        events=np.empty((0, 3), dtype=int),
        metadata={"source_format": "test"},
    )
    result = PreprocessingPipeline().run(no_event)
    assert set(result.labels.tolist()) == {0, 1}
    assert "placeholder labels" in result.summary["warning"]


def test_quick_experiment_runs(tmp_path: Path):
    result = run_experiment(
        "exp_test",
        ExperimentConfig(
            name="test",
            dataset_path=None,
            preprocessing={"epoch_start": 0.0, "epoch_end": 5.0},
            model={"model_family": "cnn-lstm"},
            output_dir=str(tmp_path),
            seed=7,
        ),
    )
    assert result.status == "succeeded"
    assert 0 <= result.metrics["accuracy"] <= 1
    assert result.folds


def test_no_event_csv_experiment_runs_with_fallback(tmp_path: Path):
    csv_path = tmp_path / "recording.csv"
    rows = ["time,ch1,ch2"]
    rows.extend(f"{index},{np.sin(index / 5):.4f},{np.cos(index / 7):.4f}" for index in range(100))
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    result = run_experiment(
        "exp_no_event",
        ExperimentConfig(
            name="no event",
            dataset_path=str(csv_path),
            output_dir=str(tmp_path),
            seed=3,
        ),
    )
    assert result.status == "succeeded"
    assert result.preprocessing_summary["warning"]


def test_ragged_csv_dataset_summary_does_not_crash(tmp_path: Path):
    csv_path = tmp_path / "ragged.csv"
    csv_path.write_text(
        "\n".join(
            [
                "time,ch1,ch2,label,subject",
                "0,0.1,0.2,rest,s1",
                "1,0.2,,task,s1",
                "2,bad,0.4,rest,s2",
                "3,0.4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    data = load_fNIRS_data(csv_path)
    summary = data.summary()
    assert summary["n_channels"] == 2
    assert summary["n_samples"] == 4
    assert summary["n_events"] == 4


def test_mat_struct_signal_dataset_summary_does_not_crash(tmp_path: Path):
    mat_path = tmp_path / "recording.mat"
    savemat(
        mat_path,
        {
            "data": {
                "dataTimeSeries": np.arange(60, dtype=np.float32).reshape(20, 3),
                "time": np.arange(20, dtype=np.float32) * 0.1,
            },
            "stim": np.asarray([[0], [1], [0], [2], [0]], dtype=np.int32),
            "fs": np.asarray([[10.0]], dtype=np.float32),
        },
    )
    data = load_fNIRS_data(mat_path)
    summary = data.summary()
    assert summary["n_channels"] == 3
    assert summary["n_samples"] == 20
    assert summary["sampling_rate"] == 10.0
    assert summary["n_events"] == 2


def test_zip_skips_unparseable_candidate_and_loads_next_file(tmp_path: Path):
    archive_path = tmp_path / "mixed.zip"
    bad_mat = tmp_path / "bad.mat"
    good_csv = tmp_path / "good.csv"
    savemat(bad_mat, {"data": {"metadata": "not a signal"}})
    good_csv.write_text("time,ch1,ch2,label\n0,0.1,0.2,rest\n1,0.2,0.3,task", encoding="utf-8")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(bad_mat, "a_bad.mat")
        archive.write(good_csv, "b_good.csv")
    data = load_fNIRS_data(archive_path)
    summary = data.summary()
    assert summary["source_format"] == "csv"
    assert summary["n_channels"] == 2
    assert summary["n_samples"] == 2


def test_same_paper_reuses_rag_directory(tmp_path: Path, monkeypatch):
    from backend import papers, services

    monkeypatch.setattr(services, "EXTRACTED_KNOWLEDGE_DIR", tmp_path / "knowledge" / "uploads" / "extracted")
    monkeypatch.setattr(services, "refresh_knowledge_base", lambda: None)
    paper_path = tmp_path / "paper.txt"
    paper_path.write_text("paper body", encoding="utf-8")
    material = PaperMaterial(
        query="read paper",
        search_query="fnirs transformer",
        candidate=PaperCandidate(title="Transformer Models for fNIRS Decoding", doi="10.1234/example"),
        workspace=str(tmp_path),
        paper_path=str(paper_path),
        paper_text="paper body",
    )
    first = finalize_paper_workflow(material, "first report")
    second = finalize_paper_workflow(material, "second report")
    paper_dirs = list((services.EXTRACTED_KNOWLEDGE_DIR / "papers").glob("*"))
    assert len([path for path in paper_dirs if path.is_dir()]) == 1
    assert Path(first.report_file) == Path(second.report_file)


def test_data_zip_rejects_unsafe_paths(tmp_path: Path):
    archive_path = tmp_path / "unsafe_data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.csv", "time,ch1\n0,1")
    try:
        load_fNIRS_data(archive_path)
    except NIRSDataError as exc:
        assert "unsafe path" in str(exc)
    else:
        raise AssertionError("unsafe data zip path should be rejected")
