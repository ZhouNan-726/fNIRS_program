<template>
  <section class="split-page">
    <div class="chat-column">
      <header class="page-header compact">
        <div>
          <h1>智能体对话</h1>
          <p>{{ displayText(store.currentStatus) }}</p>
        </div>
        <button type="button" class="ghost-button" @click="newChat">新对话</button>
      </header>

      <main ref="scroller" class="message-area">
        <article
          v-for="(message, index) in store.messages"
          :key="index"
          :class="['message-row', message.role === 'user' ? 'user' : 'assistant']"
        >
          <span class="avatar">{{ message.role === "user" ? "我" : "助" }}</span>
          <div class="message-card">
            <strong>{{ message.role === "user" ? "我" : "近红外智能体" }}</strong>
            <pre>{{ message.content }}</pre>
          </div>
        </article>
      </main>

      <form class="composer" @submit.prevent="send">
        <textarea
          v-model="input"
          :disabled="store.loading"
          placeholder="输入关于功能性近红外预处理、知识库、数据集、留一被试验证、解释或报告的问题。"
          @keydown.enter.exact.prevent="send"
        ></textarea>
        <button type="submit" class="primary-button" :disabled="!input.trim() || store.loading">
          {{ store.loading ? "生成中" : "发送" }}
        </button>
      </form>
    </div>
    <RightRail />
  </section>
</template>

<script setup>
import { nextTick, onMounted, ref } from "vue";
import { streamChat } from "../api";
import RightRail from "../components/RightRail.vue";
import { usePlatformStore } from "../stores/platform";
import { displayText } from "../formatters";

const store = usePlatformStore();
const input = ref("");
const scroller = ref(null);

function newChat() {
  store.sessionId = "";
  store.sources = [];
  store.agentSteps = [];
  store.messages = [
    {
      role: "assistant",
      content: "新的对话已开始。你可以直接提问，或让我引导数据、训练、解释和报告流程。",
    },
  ];
}

async function scrollBottom() {
  await nextTick();
  if (scroller.value) {
    scroller.value.scrollTop = scroller.value.scrollHeight;
  }
}

async function send() {
  const message = input.value.trim();
  if (!message || store.loading) return;
  input.value = "";
  store.loading = true;
  store.sources = [];
  store.agentSteps = [];
  store.currentStatus = "调度智能体正在处理";
  store.messages.push({ role: "user", content: message });
  store.messages.push({ role: "assistant", content: "" });
  const assistantIndex = store.messages.length - 1;
  await scrollBottom();
  try {
    await streamChat({ message, sessionId: store.sessionId }, async (event) => {
      if (event.event === "status") {
        store.currentStatus = displayText(event.message || store.currentStatus);
        store.pushTimeline(store.currentStatus);
      } else if (event.event === "agent_step") {
        store.agentSteps.push({
          agent: event.agent,
          action: event.action,
          detail: event.detail,
          status: event.status,
        });
      } else if (event.event === "retrieval") {
        store.sources = event.sources || [];
      } else if (event.event === "content_chunk") {
        store.messages[assistantIndex].content += event.content || "";
        await scrollBottom();
      } else if (event.event === "final") {
        if (event.content) {
          store.messages[assistantIndex].content = event.content;
        }
        store.sources = event.sources || store.sources;
        store.agentSteps = event.agent_steps || store.agentSteps;
      } else if (event.event === "session") {
        store.sessionId = event.session_id || store.sessionId;
        store.refreshChatSessions();
      } else if (event.event === "error") {
        throw new Error(event.message || "对话请求失败");
      }
    });
  } catch (error) {
    store.messages[assistantIndex].content ||= `请求失败：${error.message || String(error)}`;
  } finally {
    store.loading = false;
    store.currentStatus = "系统就绪";
    await scrollBottom();
  }
}

onMounted(() => {
  store.refreshChatSessions();
});
</script>
