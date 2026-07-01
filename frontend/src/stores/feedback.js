import { defineStore } from "pinia";

let toastId = 0;
let uploadId = 0;
let confirmationId = 0;
let confirmationResolver = null;

function clampPercent(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

export const useFeedbackStore = defineStore("feedback", {
  state: () => ({
    toasts: [],
    uploads: [],
    confirmationDialog: null,
  }),
  actions: {
    notify({ type = "info", title = "", message = "", duration = 3600 } = {}) {
      const id = ++toastId;
      this.toasts.push({ id, type, title, message });
      if (duration > 0) {
        window.setTimeout(() => this.dismissToast(id), duration);
      }
      return id;
    },
    success(message, options = {}) {
      return this.notify({ type: "success", message, ...options });
    },
    error(message, options = {}) {
      return this.notify({ type: "error", message, duration: 6000, ...options });
    },
    info(message, options = {}) {
      return this.notify({ type: "info", message, ...options });
    },
    dismissToast(id) {
      this.toasts = this.toasts.filter((toast) => toast.id !== id);
    },
    requestConfirmation({
      title = "确认操作",
      message = "",
      confirmText = "确认",
      cancelText = "取消",
      danger = false,
    } = {}) {
      if (confirmationResolver) {
        confirmationResolver(false);
      }
      const id = ++confirmationId;
      this.confirmationDialog = { id, title, message, confirmText, cancelText, danger };
      return new Promise((resolve) => {
        confirmationResolver = resolve;
      });
    },
    resolveConfirmation(accepted) {
      if (confirmationResolver) {
        confirmationResolver(Boolean(accepted));
      }
      confirmationResolver = null;
      this.confirmationDialog = null;
    },
    startUpload({ label = "上传中", fileName = "", indeterminate = false } = {}) {
      const id = ++uploadId;
      this.uploads.push({
        id,
        label,
        fileName,
        percent: 0,
        indeterminate,
        status: "running",
      });
      return id;
    },
    updateUpload(id, percent) {
      const upload = this.uploads.find((item) => item.id === id);
      if (!upload) return;
      upload.percent = clampPercent(percent);
      upload.indeterminate = false;
    },
    finishUpload(id) {
      const upload = this.uploads.find((item) => item.id === id);
      if (!upload) return;
      upload.percent = 100;
      upload.indeterminate = false;
      upload.status = "success";
      window.setTimeout(() => this.removeUpload(id), 1400);
    },
    failUpload(id) {
      const upload = this.uploads.find((item) => item.id === id);
      if (!upload) return;
      upload.status = "error";
      window.setTimeout(() => this.removeUpload(id), 3000);
    },
    removeUpload(id) {
      this.uploads = this.uploads.filter((upload) => upload.id !== id);
    },
  },
});
