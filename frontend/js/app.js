// frontend/js/app.js

// =========================
// Configuración de la URL del backend
// =========================

const API_BASE_URL = window.location.origin;

// =========================
// Referencias a elementos del DOM
// =========================

const chatContainer = document.getElementById("chat-container");
const form = document.getElementById("message-form");
const input = document.getElementById("user-input");
const sendButton = form.querySelector(".btn-send");

let isSending = false;

// =========================
// Funciones de ayuda
// =========================

function addMessage(role, text) {
  const row = document.createElement("div");
  row.classList.add("message-row");
  row.classList.add(role === "user" ? "user" : "assistant");

  const bubble = document.createElement("div");
  bubble.classList.add("message-bubble");

  const meta = document.createElement("div");
  meta.classList.add("message-meta");
  meta.textContent = role === "user" ? "Tú" : "Asistente consular";

  const body = document.createElement("div");
  body.classList.add("message-text");
  body.textContent = text;

  bubble.appendChild(meta);
  bubble.appendChild(body);
  row.appendChild(bubble);
  chatContainer.appendChild(row);

  scrollToBottom();

  return { row, bubble, meta, body };
}

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// =========================
// Eventos
// =========================

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (isSending) return;

  const question = input.value.trim();
  if (!question) return;

  addMessage("user", question);

  input.value = "";
  input.style.height = "auto";

  isSending = true;
  sendButton.disabled = true;

  const tempMsg = addMessage(
    "assistant",
    "El asistente está revisando la información..."
  );

  try {
    const response = await fetch(`${API_BASE_URL}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });

    let answerText = "";

    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      const detail =
        errorData && errorData.detail
          ? errorData.detail
          : "Ocurrió un error al procesar tu consulta.";
      answerText =
        "Lo siento, hubo un problema al procesar tu consulta:\n\n" +
        detail +
        "\n\nIntenta nuevamente en unos momentos.";
    } else {
      const data = await response.json();
      answerText = data.answer || "No se recibió respuesta del asistente.";
    }

    tempMsg.body.textContent = answerText;
  } catch (err) {
    console.error(err);
    tempMsg.body.textContent =
      "Lo siento, no pude comunicarme con el asistente. " +
      "Verifica tu conexión e inténtalo de nuevo.";
  } finally {
    isSending = false;
    sendButton.disabled = false;
    scrollToBottom();
  }
});
