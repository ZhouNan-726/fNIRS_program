<template>
  <section class="panel">
    <div class="panel-header">
      <h2>任务队列</h2>
      <button type="button" class="ghost-button" @click="store.refreshJobs">刷新</button>
    </div>
    <div v-if="!store.jobs.length" class="empty-state">暂无任务。</div>
    <div v-else class="job-list">
      <article v-for="job in store.jobs" :key="job.id" class="job-row">
        <div>
          <strong>{{ jobKindText(job.kind) }}</strong>
          <span>{{ displayText(job.message) }}</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" :style="{ width: `${Math.round(job.progress * 100)}%` }"></div>
        </div>
        <span :class="['badge', job.status]">{{ statusText(job.status) }}</span>
      </article>
    </div>
  </section>
</template>

<script setup>
import { usePlatformStore } from "../stores/platform";
import { displayText, jobKindText, statusText } from "../formatters";

const store = usePlatformStore();
</script>
