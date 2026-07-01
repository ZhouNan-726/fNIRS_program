<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1>知识库</h1>
        <p>管理检索增强生成使用的文档、片段和本地向量索引。</p>
      </div>
      <div class="button-row">
        <label :class="['file-button', uploading ? 'disabled' : '']">
          {{ uploading ? "上传中" : "上传文档" }}
          <input
            type="file"
            accept=".pdf,.docx,.doc,.md,.markdown,.txt,.text"
            :disabled="uploading"
            @change="upload"
          />
        </label>
        <button type="button" class="primary-button" :disabled="store.rebuilding" @click="rebuild">
          {{ store.rebuilding ? "重建中" : "重建索引" }}
        </button>
      </div>
    </header>

    <div class="metric-grid compact-grid">
      <MetricTile label="文档数" :value="store.knowledge?.total_documents ?? 0" />
      <MetricTile label="片段数" :value="store.knowledge?.total_chunks ?? 0" />
      <MetricTile label="向量模型" :value="store.knowledge?.embedding_model || '-'" />
    </div>

    <div class="knowledge-grid">
      <section class="panel">
        <div class="panel-header">
          <h2>文档列表</h2>
          <button type="button" class="ghost-button" @click="store.refreshDocuments">刷新</button>
        </div>
        <article
          v-for="document in store.documents"
          :key="document.id"
          :class="['list-row', store.activeDocument?.id === document.id ? 'selected' : '']"
          @click="store.loadKnowledgeDocument(document.id)"
        >
          <div>
            <strong>{{ document.title }}</strong>
            <span>{{ document.managed ? "托管文档" : "内置文档" }}</span>
          </div>
          <div class="row-actions">
            <span class="badge">{{ document.chunk_count }} 个片段</span>
            <button
              type="button"
              class="danger-button compact-action"
              :disabled="!document.managed"
              @click.stop="deleteDocument(document)"
            >
              删除
            </button>
          </div>
        </article>
      </section>

      <section class="panel editor-panel">
        <div class="panel-header">
          <h2>{{ store.activeDocument?.title || "新建文档" }}</h2>
          <button
            v-if="store.activeDocument?.managed"
            type="button"
            class="danger-button"
            @click="deleteActive"
          >
            删除
          </button>
        </div>
        <form class="document-form" @submit.prevent="submitDocument">
          <input v-model="title" :readonly="isReadOnly" placeholder="文档标题" />
          <textarea
            v-model="content"
            :readonly="isReadOnly"
            placeholder="在这里编写功能性近红外知识笔记。"
          ></textarea>
          <div class="button-row">
            <button type="submit" class="primary-button" :disabled="isReadOnly">
              {{ store.activeDocument?.managed ? "保存" : "创建" }}
            </button>
            <button
              v-if="store.activeDocument"
              type="button"
              class="ghost-button"
              @click="clearSelection"
            >
              新建
            </button>
          </div>
          <p v-if="isReadOnly" class="form-note">内置知识仅支持查看；如需扩展，请新建托管文档。</p>
        </form>

        <div v-if="store.activeDocument" class="chunk-list">
          <h3>知识片段</h3>
          <article v-for="chunk in store.activeDocument.chunks" :key="chunk.chunk_id" class="chunk-card">
            <div class="chunk-header">
              <strong>#{{ chunk.order + 1 }}</strong>
              <label class="switch-row">
                <input
                  type="checkbox"
                  :checked="chunk.enabled"
                  @change="toggleChunk(chunk, $event.target.checked)"
                />
                <span>{{ chunk.enabled ? "已启用" : "已停用" }}</span>
              </label>
            </div>
            <p>{{ chunk.content }}</p>
          </article>
        </div>
      </section>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, ref, watch } from "vue";
import MetricTile from "../components/MetricTile.vue";
import { getApiError } from "../api";
import { useFeedbackStore } from "../stores/feedback";
import { usePlatformStore } from "../stores/platform";

const store = usePlatformStore();
const feedback = useFeedbackStore();
const title = ref("");
const content = ref("");
const uploading = ref(false);
const isReadOnly = computed(() => Boolean(store.activeDocument && !store.activeDocument.managed));

async function upload(event) {
  const file = event.target.files?.[0];
  event.target.value = "";
  if (!file || uploading.value) return;
  uploading.value = true;
  const uploadId = feedback.startUpload({ label: "上传知识文档", fileName: file.name, indeterminate: true });
  try {
    await store.uploadKnowledge(file, {
      onUploadProgress(progressEvent) {
        const total = progressEvent.total || file.size || 0;
        if (total > 0) {
          feedback.updateUpload(uploadId, (progressEvent.loaded / total) * 100);
        }
      },
    });
    feedback.finishUpload(uploadId);
    feedback.success("知识文档上传完成");
  } catch (error) {
    feedback.failUpload(uploadId);
    feedback.error(getApiError(error));
  } finally {
    uploading.value = false;
  }
}

async function rebuild() {
  try {
    await store.rebuildKnowledge();
    feedback.success("知识库索引已重建");
  } catch (error) {
    feedback.error(getApiError(error));
  }
}

async function submitDocument() {
  if (!title.value.trim() || !content.value.trim()) return;
  try {
    if (store.activeDocument?.managed) {
      await store.updateKnowledge(store.activeDocument.id, { title: title.value, content: content.value });
    } else {
      await store.createKnowledge({ title: title.value, content: content.value });
      title.value = "";
      content.value = "";
    }
    feedback.success("知识文档已保存");
  } catch (error) {
    feedback.error(getApiError(error));
  }
}

function clearSelection() {
  store.activeDocument = null;
  title.value = "";
  content.value = "";
}

async function toggleChunk(chunk, enabled) {
  if (!store.activeDocument) return;
  try {
    await store.toggleChunk(store.activeDocument.id, chunk.order, enabled);
  } catch (error) {
    feedback.error(getApiError(error));
  }
}

async function deleteActive() {
  if (!store.activeDocument) return;
  await deleteDocument(store.activeDocument);
}

async function deleteDocument(document) {
  if (!document.managed) return;
  const accepted = await feedback.requestConfirmation({
    title: "删除知识文档",
    message: `确认删除知识文档「${document.title}」？`,
    confirmText: "删除",
    danger: true,
  });
  if (!accepted) return;
  try {
    await store.deleteKnowledge(document.id);
    feedback.success("知识文档已删除");
  } catch (error) {
    feedback.error(getApiError(error));
  }
}

onMounted(() => {
  store.refreshDocuments();
});

watch(
  () => store.activeDocument,
  (document) => {
    if (!document) {
      return;
    }
    title.value = document.title || "";
    content.value = (document.content || "").replace(/^# .+?\n\n/, "");
  },
);
</script>
