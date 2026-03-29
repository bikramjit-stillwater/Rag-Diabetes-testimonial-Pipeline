async function sendMessage() {
  const input = document.getElementById("user-input");
  const query = input.value.trim();
  if (!query) return;

  appendMessage("user", query);
  input.value = "";

  const response = await fetch("/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ query })
  });

  const data = await response.json();

  if (data.error) {
    appendMessage("bot", "Error: " + data.error);
    return;
  }

  appendBotResponse(data.answer, data.sources);
}

function sendPresetQuestion(question) {
  document.getElementById("user-input").value = question;
  sendMessage();
}

function appendMessage(sender, text) {
  const chatBox = document.getElementById("chat-box");
  const div = document.createElement("div");
  div.className = `message ${sender}`;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function appendBotResponse(answer, sources) {
  const chatBox = document.getElementById("chat-box");
  const div = document.createElement("div");
  div.className = "message bot";

  let html = `<div>${answer.replace(/\n/g, "<br>")}</div>`;

  if (sources && sources.length > 0) {
    html += `<div class="sources"><strong>Sources:</strong><ul>`;
    sources.forEach(src => {
      html += `<li><a href="${src.url}" target="_blank">${src.title}</a></li>`;
    });
    html += `</ul></div>`;
  }

  div.innerHTML = html;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}