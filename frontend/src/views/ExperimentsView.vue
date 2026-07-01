<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1>实验编排</h1>
        <p>创建按被试划分的本地实验，并跟踪运行任务。</p>
      </div>
      <button type="button" class="primary-button" @click="create">创建实验</button>
    </header>

    <div class="experiment-grid">
      <section class="panel">
        <h2>实验配置</h2>
        <label>
          实验名称
          <input v-model="form.name" />
        </label>
        <label>
          数据集
          <select v-model="form.dataset_id">
            <option value="">演示数据集</option>
            <option v-for="dataset in store.datasets" :key="dataset.id" :value="dataset.id">
              {{ dataset.name }}
            </option>
          </select>
        </label>
        <label>
          模型族
          <select v-model="form.model.model_family">
            <option value="fnirs-eegnet">fNIRS-EEGNet（近红外轻量网络）</option>
            <option value="cnn-lstm">CNN-LSTM（卷积循环网络）</option>
            <option value="tcn">TCN（时序卷积网络）</option>
            <option value="graph-tcn">Graph-TCN（图时序卷积网络）</option>
            <option value="hybrid-3d-cnn">Hybrid 3D CNN（混合三维卷积网络）</option>
          </select>
        </label>
        <label>
          验证策略
          <select v-model="form.validation_strategy">
            <option value="loso">LOSO（留一被试验证）</option>
            <option value="group-kfold">Group K-Fold（按被试分组交叉验证）</option>
          </select>
        </label>
        <div class="form-row">
          <label>
            折数
            <input v-model.number="form.num_folds" type="number" min="2" max="10" />
          </label>
          <label>
            随机种子
            <input v-model.number="form.seed" type="number" />
          </label>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header">
          <h2>实验列表</h2>
          <button type="button" class="ghost-button" @click="store.refreshExperiments">刷新</button>
        </div>
        <div v-if="!store.experiments.length" class="empty-state">暂无实验。</div>
        <article
          v-for="experiment in store.experiments"
          :key="experiment.id"
          :class="['list-row', store.activeExperimentId === experiment.id ? 'selected' : '']"
          @click="store.activeExperimentId = experiment.id"
        >
          <div>
            <strong>{{ experiment.name }}</strong>
            <span>{{ experiment.id }}</span>
          </div>
          <div class="row-actions">
            <span :class="['badge', experiment.status]">{{ statusText(experiment.status) }}</span>
            <button type="button" class="danger-button compact-action" @click.stop="deleteExperiment(experiment)">
              删除
            </button>
          </div>
        </article>
        <div class="button-row spaced">
          <button
            type="button"
            class="primary-button"
            :disabled="!store.activeExperimentId"
            @click="runActive"
          >
            运行
          </button>
          <button
            type="button"
            class="ghost-button"
            :disabled="!store.activeExperimentId"
            @click="explainActive"
          >
            生成解释
          </button>
        </div>
      </section>
    </div>

    <JobMonitor />
  </section>
</template>

<script setup>
import { onMounted, reactive } from "vue";
import JobMonitor from "../components/JobMonitor.vue";
import { getApiError } from "../api";
import { useFeedbackStore } from "../stores/feedback";
import { usePlatformStore } from "../stores/platform";
import { statusText } from "../formatters";

const store = usePlatformStore();
const feedback = useFeedbackStore();
const form = reactive({
  name: "快速近红外实验",
  dataset_id: "",
  preprocessing: {},
  model: { model_family: "cnn-lstm", max_epochs: 20, batch_size: 16 },
  validation_strategy: "loso",
  num_folds: 5,
  seed: 42,
});

async function create() {
  try {
    await store.createExperiment({ ...form, dataset_id: form.dataset_id || null });
    feedback.success("实验已创建");
  } catch (error) {
    feedback.error(getApiError(error));
  }
}

async function runActive() {
  try {
    await store.runExperiment(store.activeExperimentId);
    feedback.success("实验任务已提交");
  } catch (error) {
    feedback.error(getApiError(error));
  }
}

async function explainActive() {
  try {
    await store.explainExperiment(store.activeExperimentId);
    feedback.success("解释任务已提交");
  } catch (error) {
    feedback.error(getApiError(error));
  }
}

async function deleteExperiment(experiment) {
  const accepted = await feedback.requestConfirmation({
    title: "删除实验",
    message: `确认删除实验「${experiment.name}」？实验结果、解释、报告和关联任务也会删除。`,
    confirmText: "删除",
    danger: true,
  });
  if (!accepted) return;
  try {
    await store.deleteExperiment(experiment.id);
    feedback.success("实验已删除");
  } catch (error) {
    feedback.error(getApiError(error));
  }
}

onMounted(() => {
  store.refreshDatasets();
  store.refreshExperiments();
  store.refreshJobs();
});
</script>
