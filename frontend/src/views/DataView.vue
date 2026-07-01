<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1>数据管理</h1>
        <p>上传功能性近红外文件，检查通道、事件、标签和被试信息。</p>
      </div>
      <label :class="['file-button', uploading ? 'disabled' : '']">
        {{ uploading ? "上传中" : "上传数据集" }}
        <input
          type="file"
          accept=".snirf,.nirs,.mat,.csv,.zip,.json"
          :disabled="uploading"
          @change="upload"
        />
      </label>
    </header>

    <div class="two-column">
      <section class="panel">
        <div class="panel-header">
          <h2>数据集</h2>
          <button type="button" class="ghost-button" @click="store.refreshDatasets">刷新</button>
        </div>
        <div v-if="!store.datasets.length" class="empty-state">暂无数据集。</div>
        <article v-for="dataset in store.datasets" :key="dataset.id" class="list-row">
          <div>
            <strong>{{ dataset.name }}</strong>
            <span>{{ dataset.suffix }}</span>
          </div>
          <div class="row-actions">
            <span class="badge">已上传</span>
            <button type="button" class="danger-button compact-action" @click="deleteDataset(dataset)">
              删除
            </button>
          </div>
        </article>
      </section>

      <section class="panel">
        <h2>数据摘要</h2>
        <div v-if="!store.datasets.length" class="empty-state">上传数据集后查看摘要。</div>
        <div v-else class="summary-grid">
          <MetricTile label="通道数" :value="latest.summary.n_channels ?? 0" />
          <MetricTile label="采样点" :value="latest.summary.n_samples ?? 0" />
          <MetricTile label="事件数" :value="latest.summary.n_events ?? 0" />
          <MetricTile label="被试数" :value="latest.summary.subject_count ?? 0" />
          <MetricTile label="采样率" :value="latest.summary.sampling_rate ?? '-'" />
          <MetricTile label="时长（秒）" :value="latest.summary.duration_seconds ?? '-'" />
        </div>
        <pre v-if="latest">{{ formatJsonChinese(latest.summary) }}</pre>
      </section>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from "vue";
import MetricTile from "../components/MetricTile.vue";
import { getApiError } from "../api";
import { useFeedbackStore } from "../stores/feedback";
import { usePlatformStore } from "../stores/platform";
import { formatJsonChinese } from "../formatters";

const store = usePlatformStore();
const feedback = useFeedbackStore();
const latest = computed(() => store.datasets[0] || null);
const uploading = ref(false);

async function upload(event) {
  const file = event.target.files?.[0];
  event.target.value = "";
  if (!file || uploading.value) return;
  uploading.value = true;
  const uploadId = feedback.startUpload({ label: "上传数据集", fileName: file.name, indeterminate: true });
  try {
    await store.uploadDataset(file, {
      onUploadProgress(progressEvent) {
        const total = progressEvent.total || file.size || 0;
        if (total > 0) {
          feedback.updateUpload(uploadId, (progressEvent.loaded / total) * 100);
        }
      },
    });
    feedback.finishUpload(uploadId);
    feedback.success("数据集上传完成");
  } catch (error) {
    feedback.failUpload(uploadId);
    feedback.error(getApiError(error));
  } finally {
    uploading.value = false;
  }
}

async function deleteDataset(dataset) {
  const accepted = await feedback.requestConfirmation({
    title: "删除数据集",
    message: `确认删除数据集「${dataset.name}」？关联实验会保留，但会解除数据集引用。`,
    confirmText: "删除",
    danger: true,
  });
  if (!accepted) return;
  try {
    await store.deleteDataset(dataset.id);
    feedback.success("数据集已删除");
  } catch (error) {
    feedback.error(getApiError(error));
  }
}

onMounted(() => {
  store.refreshDatasets();
});
</script>
