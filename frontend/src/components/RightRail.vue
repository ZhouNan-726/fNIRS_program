<template>
  <aside class="right-rail">
    <section>
      <div class="rail-header">
        <h3>历史会话</h3>
        <button type="button" class="mini-button" @click="store.refreshChatSessions">刷新</button>
      </div>
      <div v-if="!store.chatSessions.length" class="empty-state">暂无会话。</div>
      <div v-else class="session-list">
        <div
          v-for="session in store.chatSessions"
          :key="session.id"
          :class="['session-item', store.sessionId === session.id ? 'selected' : '']"
        >
          <button type="button" class="session-button" @click="store.loadChatSession(session.id)">
            <strong>{{ session.title }}</strong>
            <span>{{ formatDate(session.updated_at) }}</span>
          </button>
          <button type="button" class="icon-danger-button" title="删除会话" @click="deleteSession(session)">
            ×
          </button>
        </div>
      </div>
    </section>

    <section>
      <h3>智能体步骤</h3>
      <div v-if="!store.agentSteps.length" class="empty-state">暂无步骤。</div>
      <ol v-else class="step-list">
        <li v-for="(step, index) in store.agentSteps" :key="index">
          <strong>{{ agentText(step.agent) }}</strong>
          <span>{{ actionText(step.action) }}</span>
          <small>{{ displayText(step.detail) }}</small>
        </li>
      </ol>
    </section>

    <section>
      <h3>检索来源</h3>
      <div v-if="!store.sources.length" class="empty-state">暂无来源。</div>
      <div v-else class="source-list">
        <article v-for="(source, index) in store.sources" :key="index">
          <strong>{{ source.title }}</strong>
          <span>第 {{ index + 1 }} 个来源</span>
          <small>相关度 {{ source.score }}</small>
          <p>{{ source.snippet }}</p>
        </article>
      </div>
    </section>
  </aside>
</template>

<script setup>
import { getApiError } from "../api";
import { useFeedbackStore } from "../stores/feedback";
import { usePlatformStore } from "../stores/platform";
import { actionText, agentText, displayText, formatDate } from "../formatters";

const store = usePlatformStore();
const feedback = useFeedbackStore();

async function deleteSession(session) {
  const accepted = await feedback.requestConfirmation({
    title: "删除会话",
    message: `确认删除会话「${session.title}」？`,
    confirmText: "删除",
    danger: true,
  });
  if (!accepted) return;
  try {
    await store.deleteChatSession(session.id);
    feedback.success("会话已删除");
  } catch (error) {
    feedback.error(getApiError(error));
  }
}
</script>
