<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1>运行设置</h1>
        <p>查看和调整后端使用的本地运行配置。</p>
      </div>
      <div class="button-row">
        <button type="button" class="ghost-button" :disabled="store.savingSettings" @click="refresh">刷新</button>
        <button type="submit" form="runtime-settings-form" class="primary-button" :disabled="!canSave">
          {{ store.savingSettings ? "保存中" : "保存配置" }}
        </button>
      </div>
    </header>

    <form id="runtime-settings-form" class="panel settings-form" @submit.prevent="save">
      <h2>本地运行环境</h2>
      <div class="settings-grid">
        <label>
          Ollama 地址
          <input :value="store.health?.ollama_base_url || '未配置'" readonly />
        </label>
        <label>
          对话模型
          <input v-model.trim="form.chat_model" autocomplete="off" placeholder="qwen3:8b" />
        </label>
        <label>
          向量模型
          <input v-model.trim="form.embedding_model" autocomplete="off" placeholder="qwen3-embedding:8b" />
        </label>
        <label>
          本地数据库
          <input :value="store.health?.database_path || '未配置'" readonly />
        </label>
        <label>
          本地向量库
          <input :value="store.health?.vector_store_path || '未配置'" readonly />
        </label>
      </div>
    </form>

    <section class="panel">
      <h2>状态详情</h2>
      <div v-if="store.health" class="summary-grid">
        <MetricTile label="接口状态" :value="statusText(store.health.api_status)" />
        <MetricTile label="模型服务" :value="statusText(store.health.ollama_status)" />
        <MetricTile label="对话模型" :value="statusText(store.health.chat_model_status)" />
        <MetricTile label="向量模型" :value="statusText(store.health.embedding_model_status)" />
      </div>
    </section>
  </section>
</template>

<script setup>
import { computed, onMounted, reactive, watch } from "vue";
import MetricTile from "../components/MetricTile.vue";
import { getApiError } from "../api";
import { useFeedbackStore } from "../stores/feedback";
import { usePlatformStore } from "../stores/platform";
import { statusText } from "../formatters";

const store = usePlatformStore();
const feedback = useFeedbackStore();
const form = reactive({
  chat_model: "",
  embedding_model: "",
});

const canSave = computed(() => {
  if (store.savingSettings) return false;
  if (!form.chat_model.trim() || !form.embedding_model.trim()) return false;
  return (
    form.chat_model.trim() !== (store.health?.chat_model || "") ||
    form.embedding_model.trim() !== (store.health?.embedding_model || "")
  );
});

function syncForm() {
  form.chat_model = store.health?.chat_model || "";
  form.embedding_model = store.health?.embedding_model || "";
}

async function refresh() {
  await store.refreshHealth();
  syncForm();
}

async function save() {
  if (!canSave.value) return;
  try {
    await store.updateRuntimeConfig({
      chat_model: form.chat_model,
      embedding_model: form.embedding_model,
    });
    syncForm();
    feedback.success("模型配置已保存");
  } catch (error) {
    feedback.error(getApiError(error));
  }
}

watch(() => store.health, syncForm, { immediate: true });
onMounted(refresh);
</script>
