import { defineStore } from "pinia";
import { api } from "../api";

export const usePlatformStore = defineStore("platform", {
  state: () => ({
    health: null,
    knowledge: null,
    documents: [],
    activeDocument: null,
    datasets: [],
    experiments: [],
    jobs: [],
    activeExperimentId: "",
    sessionId: "",
    chatSessions: [],
    messages: [
      {
        role: "assistant",
        content: "你好，我是功能性近红外深度学习多智能体平台。你可以问知识库问题，也可以让我引导数据、实验、解释和报告流程。",
      },
    ],
    sources: [],
    agentSteps: [],
    timeline: [],
    currentStatus: "系统就绪",
    loading: false,
    uploading: false,
    rebuilding: false,
    runningExperiment: false,
    savingSettings: false,
  }),
  getters: {
    busy: (state) => state.loading || state.uploading || state.rebuilding || state.runningExperiment || state.savingSettings,
    activeExperiment: (state) => state.experiments.find((item) => item.id === state.activeExperimentId) || state.experiments[0] || null,
  },
  actions: {
    pushTimeline(message) {
      if (!message) return;
      this.timeline.push(message);
      if (this.timeline.length > 80) {
        this.timeline = this.timeline.slice(-80);
      }
    },
    async refreshAll() {
      await Promise.allSettled([
        this.refreshHealth(),
        this.refreshKnowledge(),
        this.refreshDocuments(),
        this.refreshDatasets(),
        this.refreshExperiments(),
        this.refreshJobs(),
        this.refreshChatSessions(),
      ]);
    },
    async refreshHealth() {
      const { data } = await api.get("/health");
      this.health = data;
      return data;
    },
    async updateRuntimeConfig(payload) {
      this.savingSettings = true;
      try {
        const { data } = await api.put("/settings", payload);
        this.health = data;
        return data;
      } finally {
        this.savingSettings = false;
      }
    },
    async refreshKnowledge() {
      const { data } = await api.get("/knowledge");
      this.knowledge = data;
      return data;
    },
    async refreshChatSessions() {
      const { data } = await api.get("/chat/sessions");
      this.chatSessions = data.sessions || [];
      return data;
    },
    async loadChatSession(sessionId) {
      const { data } = await api.get(`/chat/sessions/${sessionId}`);
      this.sessionId = data.id;
      this.messages = data.messages?.length ? data.messages : this.messages;
      this.sources = data.sources || [];
      this.agentSteps = data.agent_steps || [];
      return data;
    },
    async deleteChatSession(sessionId) {
      await api.delete(`/chat/sessions/${sessionId}`);
      if (this.sessionId === sessionId) {
        this.sessionId = "";
        this.sources = [];
        this.agentSteps = [];
        this.messages = [
          {
            role: "assistant",
            content: "新的对话已开始。你可以直接提问，或让我引导数据、训练、解释和报告流程。",
          },
        ];
      }
      await this.refreshChatSessions();
    },
    async refreshDocuments() {
      const { data } = await api.get("/knowledge/documents");
      this.documents = data.documents || [];
      this.knowledge = data.knowledge || this.knowledge;
      return data;
    },
    async rebuildKnowledge() {
      this.rebuilding = true;
      try {
        const { data } = await api.post("/knowledge/refresh");
        this.knowledge = data;
        await this.refreshDocuments();
        return data;
      } finally {
        this.rebuilding = false;
      }
    },
    async uploadKnowledge(file, options = {}) {
      const formData = new FormData();
      formData.append("file", file);
      const { data } = await api.post("/knowledge/upload", formData, {
        onUploadProgress: options.onUploadProgress,
      });
      this.knowledge = data.knowledge;
      await this.refreshDocuments();
      return data;
    },
    async createKnowledge(payload) {
      const { data } = await api.post("/knowledge/documents", payload);
      this.activeDocument = data;
      await this.refreshDocuments();
      return data;
    },
    async updateKnowledge(documentId, payload) {
      const { data } = await api.put(`/knowledge/documents/${documentId}`, payload);
      this.activeDocument = data;
      await this.refreshDocuments();
      return data;
    },
    async loadKnowledgeDocument(documentId) {
      const { data } = await api.get(`/knowledge/documents/${documentId}`);
      this.activeDocument = data;
      return data;
    },
    async toggleChunk(documentId, order, enabled) {
      const { data } = await api.patch(`/knowledge/documents/${documentId}/chunks/${order}`, { enabled });
      this.activeDocument = data;
      await this.refreshDocuments();
      return data;
    },
    async deleteKnowledge(documentId) {
      const { data } = await api.delete(`/knowledge/documents/${documentId}`);
      this.knowledge = data;
      this.activeDocument = null;
      await this.refreshDocuments();
      return data;
    },
    async refreshDatasets() {
      const { data } = await api.get("/datasets");
      this.datasets = data.datasets || [];
      return data;
    },
    async uploadDataset(file, options = {}) {
      const formData = new FormData();
      formData.append("file", file);
      const { data } = await api.post("/datasets/upload", formData, {
        onUploadProgress: options.onUploadProgress,
      });
      await this.refreshDatasets();
      return data;
    },
    async deleteDataset(datasetId) {
      await api.delete(`/datasets/${datasetId}`);
      await Promise.allSettled([this.refreshDatasets(), this.refreshExperiments()]);
    },
    async refreshExperiments() {
      const { data } = await api.get("/experiments");
      this.experiments = data.experiments || [];
      if (this.activeExperimentId && !this.experiments.some((item) => item.id === this.activeExperimentId)) {
        this.activeExperimentId = "";
      }
      if (!this.activeExperimentId && this.experiments[0]) {
        this.activeExperimentId = this.experiments[0].id;
      }
      return data;
    },
    async createExperiment(payload) {
      const { data } = await api.post("/experiments", payload);
      this.activeExperimentId = data.id;
      await this.refreshExperiments();
      return data;
    },
    async runExperiment(experimentId) {
      const { data } = await api.post(`/experiments/${experimentId}/run`);
      await this.refreshJobs();
      return data;
    },
    async explainExperiment(experimentId) {
      const { data } = await api.post(`/experiments/${experimentId}/explain`);
      await this.refreshJobs();
      return data;
    },
    async deleteExperiment(experimentId) {
      await api.delete(`/experiments/${experimentId}`);
      if (this.activeExperimentId === experimentId) {
        this.activeExperimentId = "";
      }
      await Promise.allSettled([this.refreshExperiments(), this.refreshJobs()]);
    },
    async deleteExperimentResults(experimentId) {
      const { data } = await api.delete(`/experiments/${experimentId}/results`);
      await this.refreshExperiments();
      return data;
    },
    async refreshJobs() {
      const { data } = await api.get("/jobs");
      this.jobs = data || [];
      return data;
    },
    async refreshExperimentResults(experimentId) {
      const { data } = await api.get(`/experiments/${experimentId}/results`);
      await this.refreshExperiments();
      return data;
    },
  },
});
