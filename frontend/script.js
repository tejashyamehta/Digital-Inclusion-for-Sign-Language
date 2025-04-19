// Update predictions every second
function fetchPrediction() {
  fetch("/get_prediction")
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("char").textContent = data.char || "-"
      document.getElementById("sentence").textContent = data.sentence || ""
      document.getElementById("sug1").textContent = data.suggestions[0] || ""
      document.getElementById("sug2").textContent = data.suggestions[1] || ""
      document.getElementById("sug3").textContent = data.suggestions[2] || ""
      document.getElementById("sug4").textContent = data.suggestions[3] || ""
    })
    .catch((err) => console.error("Prediction fetch error:", err))
}

// Speak the sentence
document.getElementById("speak").addEventListener("click", () => {
  fetch("/speak", { method: "POST" })
})

// Clear the sentence and UI
document
  .getElementById("clear")
  .addEventListener("click", () => {
    fetch("/clear", { method: "POST" }).then(() => {
      document.getElementById("char").textContent = "-"
      document.getElementById("sentence").textContent = ""
      ;["sug1", "sug2", "sug3", "sug4"].forEach((id) => {
        document.getElementById(id).textContent = ""
      })
    })
  })

// Suggestion buttons: add selected word to sentence
;["sug1", "sug2", "sug3", "sug4"].forEach((id) => {
  document.getElementById(id).addEventListener("click", () => {
    const word = document.getElementById(id).textContent
    if (word) {
      fetch("/get_prediction")
        .then((response) => response.json())
        .then((data) => {
          const sentence = data.sentence.trim()
          const parts = sentence.split(" ")
          parts[parts.length - 1] = word
          const updated = parts.join(" ") + " "
          fetch("/clear", { method: "POST" }).then(() => {
            document.getElementById("sentence").textContent = updated
            // Update on backend too (optional)
          })
        })
    }
  })
})

// Start polling predictions
setInterval(fetchPrediction, 2000)
