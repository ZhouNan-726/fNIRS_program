<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1>结果分析</h1>
        <p>查看指标、折次表现、解释结果和实验报告。</p>
      </div>
      <div class="button-row">
        <button type="button" class="primary-button" :disabled="!active" @click="refresh">
          刷新
        </button>
        <button type="button" class="danger-button" :disabled="!active?.result && !active?.explanation" @click="deleteResults">
          删除结果
        </button>
      </div>
    </header>

    <div class="results-grid">
      <section class="panel">
        <h2>选择实验</h2>
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
            <button
              type="button"
              class="danger-button compact-action"
              :disabled="!experiment.result && !experiment.explanation"
              @click.stop="deleteResultsFor(experiment)"
            >
              删除结果
            </button>
          </div>
        </article>
      </section>

      <section class="panel">
        <h2>核心指标</h2>
        <div v-if="!active?.result" class="empty-state">暂无结果。</div>
        <div v-else>
          <div class="summary-grid">
            <MetricTile label="准确率" :value="active.result.metrics?.accuracy ?? '-'" />
            <MetricTile label="样本数" :value="active.result.metrics?.n_samples ?? 0" />
            <MetricTile label="折次数" :value="active.result.folds?.length ?? 0" />
          </div>
          <pre>{{ formatJsonChinese(active.result.metrics) }}</pre>
        </div>
      </section>

      <section class="panel">
        <h2>折次结果</h2>
        <div v-if="!active?.result?.folds?.length" class="empty-state">暂无折次结果。</div>
          <article v-for="fold in active.result.folds" v-else :key="fold.fold_name" class="list-row">
          <div>
            <strong>{{ foldNameText(fold.fold_name) }}</strong>
            <span>训练 {{ fold.train_size }} / 验证 {{ fold.val_size }}</span>
          </div>
          <span class="badge succeeded">{{ fold.accuracy }}</span>
        </article>
      </section>

      <section class="panel">
        <div class="panel-header">
          <h2>解释结果</h2>
          <button type="button" class="ghost-button" :disabled="!active" @click="explain">生成</button>
        </div>
        <div v-if="!active?.explanation" class="empty-state">暂无解释结果。</div>
        <div v-else class="source-list">
          <article v-for="item in active.explanation.top_channels" :key="item.channel">
            <strong>{{ item.channel }}</strong>
            <span>重要性 {{ item.importance }}</span>
          </article>
        </div>
        <a
          v-if="active?.result"
          class="primary-button download-link"
          :href="`/api/reports/${active.id}/download`"
          target="_blank"
          rel="noreferrer"
        >
          下载报告
        </a>
      </section>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted } from "vue";
import MetricTile from "../components/MetricTile.vue";
import { usePlatformStore } from "../stores/platform";
import { useFeedbackStore } from "../stores/feedback";
import { getApiError } from "../api";
import { foldNameText, formatJsonChinese, statusText } from "../formatters";

const store = usePlatformStore();
const feedback = useFeedbackStore();
const active = computed(() => store.activeExperiment);

async function refresh() {
  try {
    if (active.value) {
      await store.refreshExperimentResults(active.value.id);
    }
    await store.refreshExperiments();
    feedback.success("结果已刷新");
  } catch (error) {
    feedback.error(getApiError(error));
  }
}

async function explain() {
  if (!active.value) return;
  try {
    await store.explainExperiment(active.value.id);
    feedback.success("解释任务已提交");
  } catch (error) {
    feedback.error(getApiError(error));
  }
}

async function deleteResults() {
  if (!active.value) return;
  await deleteResultsFor(active.value);
}

async function deleteResultsFor(experiment) {
  const accepted = await feedback.requestConfirmation({
    title: "删除结果",
    message: `确认删除实验「${experiment.name}」的结果、解释和报告？实验配置会保留。`,
    confirmText: "删除",
    danger: true,
  });
  if (!accepted) return;
  try {
    await store.deleteExperimentResults(experiment.id);
    feedback.success("实验结果已删除");
  } catch (error) {
    feedback.error(getApiError(error));
  }
}

onMounted(() => {
  store.refreshExperiments();
});
</script>
