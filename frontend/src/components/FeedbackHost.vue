<template>
  <Teleport to="body">
    <div v-if="feedback.toasts.length" class="feedback-toast-stack" aria-live="polite">
      <article v-for="toast in feedback.toasts" :key="toast.id" :class="['feedback-toast', toast.type]">
        <span class="feedback-toast-mark">{{ toastMark(toast.type) }}</span>
        <div class="feedback-toast-body">
          <strong v-if="toast.title">{{ toast.title }}</strong>
          <p>{{ toast.message }}</p>
        </div>
        <button type="button" class="feedback-close" aria-label="关闭提示" @click="feedback.dismissToast(toast.id)">
          x
        </button>
      </article>
    </div>

    <div v-if="feedback.uploads.length" class="feedback-upload-stack" aria-live="polite">
      <article v-for="upload in feedback.uploads" :key="upload.id" :class="['feedback-upload', upload.status]">
        <div class="feedback-upload-header">
          <div>
            <strong>{{ upload.label }}</strong>
            <span v-if="upload.fileName">{{ upload.fileName }}</span>
          </div>
          <small>{{ uploadStatus(upload) }}</small>
        </div>
        <div :class="['feedback-progress-track', upload.indeterminate ? 'indeterminate' : '']">
          <span class="feedback-progress-fill" :style="{ width: `${upload.percent}%` }"></span>
        </div>
      </article>
    </div>

    <div
      v-if="feedback.confirmationDialog"
      class="feedback-modal-backdrop"
      @click.self="feedback.resolveConfirmation(false)"
    >
      <section class="feedback-dialog" role="dialog" aria-modal="true" aria-labelledby="feedback-dialog-title">
        <h2 id="feedback-dialog-title">{{ feedback.confirmationDialog.title }}</h2>
        <p>{{ feedback.confirmationDialog.message }}</p>
        <div class="button-row feedback-dialog-actions">
          <button type="button" class="ghost-button" @click="feedback.resolveConfirmation(false)">
            {{ feedback.confirmationDialog.cancelText }}
          </button>
          <button
            type="button"
            :class="feedback.confirmationDialog.danger ? 'danger-button' : 'primary-button'"
            @click="feedback.resolveConfirmation(true)"
          >
            {{ feedback.confirmationDialog.confirmText }}
          </button>
        </div>
      </section>
    </div>
  </Teleport>
</template>

<script setup>
import { useFeedbackStore } from "../stores/feedback";

const feedback = useFeedbackStore();

function toastMark(type) {
  if (type === "success") return "✓";
  if (type === "error") return "!";
  return "i";
}

function uploadStatus(upload) {
  if (upload.status === "success") return "完成";
  if (upload.status === "error") return "失败";
  if (upload.indeterminate) return "上传中";
  return `${upload.percent}%`;
}
</script>
