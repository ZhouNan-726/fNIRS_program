<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1>工作台总览</h1>
        <p>查看本地服务、知识库、数据、实验和任务的运行状态。</p>
      </div>
      <button type="button" class="primary-button" @click="store.refreshAll">刷新</button>
    </header>

    <div class="metric-grid">
      <MetricTile label="本地模型" :value="statusText(store.health?.ollama_status)" :note="statusText(store.health?.chat_model_status)" />
      <MetricTile label="知识文档" :value="store.knowledge?.total_documents ?? 0" />
      <MetricTile label="知识片段" :value="store.knowledge?.total_chunks ?? 0" />
      <MetricTile label="数据集" :value="store.datasets.length" />
      <MetricTile label="实验数" :value="store.experiments.length" />
      <MetricTile label="任务数" :value="store.jobs.length" />
    </div>

    <div class="two-column">
      <section class="panel">
        <div class="panel-header">
          <h2>最近实验</h2>
          <RouterLink class="text-link" to="/experiments">打开</RouterLink>
        </div>
        <div v-if="!store.experiments.length" class="empty-state">暂无实验。</div>
        <article v-for="experiment in store.experiments.slice(0, 5)" :key="experiment.id" class="list-row">
          <div>
            <strong>{{ experiment.name }}</strong>
            <span>{{ experiment.id }}</span>
          </div>
          <span :class="['badge', experiment.status]">{{ statusText(experiment.status) }}</span>
        </article>
      </section>

      <JobMonitor />
    </div>
  </section>
</template>

<script setup>
import { RouterLink } from "vue-router";
import MetricTile from "../components/MetricTile.vue";
import JobMonitor from "../components/JobMonitor.vue";
import { usePlatformStore } from "../stores/platform";
import { statusText } from "../formatters";

const store = usePlatformStore();
</script>
