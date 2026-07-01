import { createRouter, createWebHistory } from "vue-router";
import DashboardView from "../views/DashboardView.vue";
import ChatView from "../views/ChatView.vue";
import KnowledgeView from "../views/KnowledgeView.vue";
import DataView from "../views/DataView.vue";
import ExperimentsView from "../views/ExperimentsView.vue";
import ResultsView from "../views/ResultsView.vue";
import SettingsView from "../views/SettingsView.vue";

const routes = [
  { path: "/", name: "dashboard", component: DashboardView },
  { path: "/chat", name: "chat", component: ChatView },
  { path: "/knowledge", name: "knowledge", component: KnowledgeView },
  { path: "/data", name: "data", component: DataView },
  { path: "/experiments", name: "experiments", component: ExperimentsView },
  { path: "/results", name: "results", component: ResultsView },
  { path: "/settings", name: "settings", component: SettingsView },
];

export default createRouter({
  history: createWebHistory(),
  routes,
});

