<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
    <title>Interactive Chat</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400;1,700&display=swap" rel="stylesheet">
</head>
<body>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz" crossorigin="anonymous"></script>
    <div id="context-options" class="context-box d-none">
      <div class="form-check">
          <input class="form-check-input" type="radio" name="contextOption" id="contextNo" value="no" >
          <label class="form-check-label" for="contextNo">No Context</label>
      </div>
      <div class="form-check">
          <input class="form-check-input" type="radio" name="contextOption" id="contextYes" value="yes"checked>
          <label class="form-check-label" for="contextYes">Add Context</label>
      </div>
      <textarea class="form-control mt-2 d-block" id="context-input" rows="3" placeholder="Enter context here..."></textarea>
      <button class="close-button" id="close-button">Close</button>
  </div>

  <div class="chat-container">
      <div class="chat-box" id="chat-box"></div>
      <div class="chat-input-container mb-3">
          <span class="material-symbols-outlined" id="context-toggle">quiz</span>
          <input class="ms-2" type="text" id="chat-input" placeholder="Message Medical Chat" />
          <button class="send-button" id="send-button">
              <span class="material-symbols-outlined bg-dark p-2 text-white rounded-circle fs-5 fw-bold">arrow_upward</span>
          </button>
      </div>
  </div>

    {{-- </div><div id="nav-bar">
        <input id="nav-toggle" type="checkbox" />
        <div id="nav-header">
          <a href="https://codepen.io" id="nav-title" target="_blank">History</a><label for="nav-toggle"><span id="nav-toggle-burger"></span></label>
          <hr />
        </div>
        <div id="nav-content">
          <div class="nav-button ms-4">
            question1
          </div>
          <div class="nav-button ms-4">
            question2
          </div>
          <div class="nav-button ms-4">
            question3
          </div>
          <hr />
          <div class="nav-button ms-4">
            question4
          </div>
          <div class="nav-button ms-4">
            question5
          </div>
          <div class="nav-button ms-4">
            question6
          </div>
          <div class="nav-button">
          </div>
          <hr />
         
          <div id="nav-content-highlight"></div>
        </div>
        <input id="nav-footer-toggle" type="checkbox" />
        <div id="nav-footer">
          <div id="nav-footer-heading">
            <div id="nav-footer-avatar">
              <img src="https://gravatar.com/avatar/4474ca42d303761c2901fa819c4f2547" />
            </div>
            <div id="nav-footer-titlebox">
              <a href="https://codepen.io/uahnbu/pens/public" id="nav-footer-title" target="_blank">uahnbu</a><span id="nav-footer-subtitle">Admin</span>
            </div>
            <label for="nav-footer-toggle"><i class="fas fa-caret-up"></i></label>
          </div>
          <div id="nav-footer-content">
            <Lorem>ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</Lorem>
          </div>
        </div>
      </div> --}}

    


</body>
    <style>
      body {
    background: #f0f0f0;
    margin: 0;
    font-family: Arial, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    align-items: center;
    background-image: url('images/3.jpg');
    background-size: cover;
    background-position: center right; /* Adjusted to move the image slightly to the right */
    width: 100%;
}



        .chat-container {
            width: 100%;
            max-width: 700px;
            margin: 0 auto;
            padding: 10px;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
        }

        .chat-box {
            /*background: #f0f0f0;*/

            padding: 10px;
            max-height: 550px;
            overflow-y: auto;
            margin-bottom: 20px;
            flex-grow: 1;
            border: none;
        }

        .chat-message {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 10px;
            background: #f8f8f8;
        }

        .chat-message.user {
            background: #d1e7dd;
            justify-content: flex-end;
        }

        .chat-message p {
            margin: 0;
        }

        .chat-input-container {
            display: flex;
            align-items: center;
            width: 100%;
            background: #f8f8f8;
            border-radius: 25px;
            padding: 10px 20px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        .chat-input-container input {
            flex: 1;
            border: none;
            outline: none;
            background: none;
            font-size: 16px;
            color: #333;
        }

        .chat-input-container .send-button {
            background: none;
            border: none;
            outline: none;
            cursor: pointer;
        }

        .send-button:hover{
            opacity: 80%;
        }

        .chat-input-container .send-button img {
            width: 20px;
            height: 20px;
        }

        :root {
  --background: #fefefe;
  --navbar-width: 256px;
  --navbar-width-min: 80px;
  --navbar-dark-primary: #18283b;
  --navbar-dark-secondary: #2c3e50;
  --navbar-light-primary: #f5f6fa;
  --navbar-light-secondary: #8392a5;
}



#nav-toggle:checked ~ #nav-header {
  width: calc(var(--navbar-width-min) - 16px);
}
#nav-toggle:checked ~ #nav-content, #nav-toggle:checked ~ #nav-footer {
  width: var(--navbar-width-min);
}
#nav-toggle:checked ~ #nav-header #nav-title {
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.1s;
}
#nav-toggle:checked ~ #nav-header label[for=nav-toggle] {
  left: calc(50% - 8px);
  transform: translate(-50%);
}
#nav-toggle:checked ~ #nav-header #nav-toggle-burger {
  background: var(--navbar-light-primary);
}
#nav-toggle:checked ~ #nav-header #nav-toggle-burger:before, #nav-toggle:checked ~ #nav-header #nav-toggle-burger::after {
  width: 16px;
  background: var(--navbar-light-secondary);
  transform: translate(0, 0) rotate(0deg);
}
#nav-toggle:checked ~ #nav-content .nav-button span {
  opacity: 0;
  transition: opacity 0.1s;
}
#nav-toggle:checked ~ #nav-content .nav-button .fas {
  min-width: calc(100% - 16px);
}
#nav-toggle:checked ~ #nav-footer #nav-footer-avatar {
  margin-left: 0;
  left: 50%;
  transform: translate(-50%);
}
#nav-toggle:checked ~ #nav-footer #nav-footer-titlebox, #nav-toggle:checked ~ #nav-footer label[for=nav-footer-toggle] {
  opacity: 0;
  transition: opacity 0.1s;
  pointer-events: none;
}

#nav-bar {
  position: absolute;
  left: 1vw;
  top: 1vw;
  height: calc(100% - 2vw);
  background: var(--navbar-dark-primary);
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  color: var(--navbar-light-primary);
  font-family: Verdana, Geneva, Tahoma, sans-serif;
  overflow: hidden;
  user-select: none;
}
#nav-bar hr {
  margin: 0;
  position: relative;
  left: 16px;
  width: calc(100% - 32px);
  border: none;
  border-top: solid 1px var(--navbar-dark-secondary);
}
#nav-bar a {
  color: inherit;
  text-decoration: inherit;
}
#nav-bar input[type=checkbox] {
  display: none;
}

#nav-header {
  position: relative;
  width: var(--navbar-width);
  left: 16px;
  width: calc(var(--navbar-width) - 16px);
  min-height: 80px;
  background: var(--navbar-dark-primary);
  border-radius: 16px;
  z-index: 2;
  display: flex;
  align-items: center;
  transition: width 0.2s;
}
#nav-header hr {
  position: absolute;
  bottom: 0;
}

#nav-title {
  font-size: 1.5rem;
  transition: opacity 1s;
}

label[for=nav-toggle] {
  position: absolute;
  right: 0;
  width: 3rem;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}

#nav-toggle-burger {
  position: relative;
  width: 16px;
  height: 2px;
  background: var(--navbar-dark-primary);
  border-radius: 99px;
  transition: background 0.2s;
}
#nav-toggle-burger:before, #nav-toggle-burger:after {
  content: "";
  position: absolute;
  top: -6px;
  width: 10px;
  height: 2px;
  background: var(--navbar-light-primary);
  border-radius: 99px;
  transform: translate(2px, 8px) rotate(30deg);
  transition: 0.2s;
}
#nav-toggle-burger:after {
  top: 6px;
  transform: translate(2px, -8px) rotate(-30deg);
}

#nav-content {
  margin: -16px 0;
  padding: 16px 0;
  position: relative;
  flex: 1;
  width: var(--navbar-width);
  background: var(--navbar-dark-primary);
  box-shadow: 0 0 0 16px var(--navbar-dark-primary);
  direction: rtl;
  overflow-x: hidden;
  transition: width 0.2s;
}
#nav-content::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
#nav-content::-webkit-scrollbar-thumb {
  border-radius: 99px;
  background-color: #D62929;
}
#nav-content::-webkit-scrollbar-button {
  height: 16px;
}

#nav-content-highlight {
  position: absolute;
  left: 16px;
  top: -70px;
  width: calc(100% - 16px);
  height: 54px;
  background: var(--background);
  background-attachment: fixed;
  border-radius: 16px 0 0 16px;
  transition: top 0.2s;
}
#nav-content-highlight:before, #nav-content-highlight:after {
  content: "";
  position: absolute;
  right: 0;
  bottom: 100%;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  box-shadow: 16px 16px var(--background);
}
#nav-content-highlight:after {
  top: 100%;
  box-shadow: 16px -16px var(--background);
}
#context-toggle{
  cursor: pointer;
}
.nav-button {
  position: relative;
  margin-left: 16px;
  height: 54px;
  display: flex;
  align-items: center;
  color: var(--navbar-light-secondary);
  direction: ltr;
  cursor: pointer;
  z-index: 1;
  transition: color 0.2s;
}
.nav-button span {
  transition: opacity 1s;
}
.nav-button .fas {
  transition: min-width 0.2s;
}
.nav-button:nth-of-type(1):hover {
  color: var(--navbar-dark-primary);
}
.nav-button:nth-of-type(1):hover ~ #nav-content-highlight {
  top: 16px;
}
.nav-button:nth-of-type(2):hover {
  color: var(--navbar-dark-primary);
}
.nav-button:nth-of-type(2):hover ~ #nav-content-highlight {
  top: 70px;
}
.nav-button:nth-of-type(3):hover {
  color: var(--navbar-dark-primary);
}
.nav-button:nth-of-type(3):hover ~ #nav-content-highlight {
  top: 124px;
}
.nav-button:nth-of-type(4):hover {
  color: var(--navbar-dark-primary);
}
.nav-button:nth-of-type(4):hover ~ #nav-content-highlight {
  top: 178px;
}
.nav-button:nth-of-type(5):hover {
  color: var(--navbar-dark-primary);
}
.nav-button:nth-of-type(5):hover ~ #nav-content-highlight {
  top: 232px;
}
.nav-button:nth-of-type(6):hover {
  color: var(--navbar-dark-primary);
}
.nav-button:nth-of-type(6):hover ~ #nav-content-highlight {
  top: 286px;
}
.nav-button:nth-of-type(7):hover {
  color: var(--navbar-dark-primary);
}
.nav-button:nth-of-type(7):hover ~ #nav-content-highlight {
  top: 340px;
}
.nav-button:nth-of-type(8):hover {
  color: var(--navbar-dark-primary);
}
.nav-button:nth-of-type(8):hover ~ #nav-content-highlight {
  top: 394px;
}

#nav-bar .fas {
  min-width: 3rem;
  text-align: center;
}

#nav-footer {
  position: relative;
  width: var(--navbar-width);
  height: 54px;
  background: var(--navbar-dark-secondary);
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  z-index: 2;
  transition: width 0.2s, height 0.2s;
}
#context-input{
  font-size: 18px;
  font-family: "Amiri", serif;
  font-weight: 700;
  font-style: normal;
}

#nav-footer-heading {
  position: relative;
  width: 100%;
  height: 54px;
  display: flex;
  align-items: center;
}

#nav-footer-avatar {
  position: relative;
  margin: 11px 0 11px 16px;
  left: 0;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  overflow: hidden;
  transform: translate(0);
  transition: 0.2s;
}
#nav-footer-avatar img {
  height: 100%;
}

#nav-footer-titlebox {
  position: relative;
  margin-left: 16px;
  width: 10px;
  display: flex;
  flex-direction: column;
  transition: opacity 1s;
}

#nav-footer-subtitle {
  color: var(--navbar-light-secondary);
  font-size: 0.6rem;
}

#nav-toggle:not(:checked) ~ #nav-footer-toggle:checked + #nav-footer {
  height: 30%;
  min-height: 54px;
}
#nav-toggle:not(:checked) ~ #nav-footer-toggle:checked + #nav-footer label[for=nav-footer-toggle] {
  transform: rotate(180deg);
}

label[for=nav-footer-toggle] {
  position: absolute;
  right: 0;
  width: 3rem;
  height: 100%;
  display: flex;
  align-items: center;
  cursor: pointer;
  transition: transform 0.2s, opacity 0.2s;
}

#nav-footer-content {
  margin: 0 16px 16px 16px;
  border-top: solid 1px var(--navbar-light-secondary);
  padding: 16px 0;
  color: var(--navbar-light-secondary);
  font-size: 0.8rem;
  overflow: auto;
}
#nav-footer-content::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
#nav-footer-content::-webkit-scrollbar-thumb {
  border-radius: 99px;
  background-color: #D62929;
}

#context-options {
    position: absolute;
    top: 250px;
    right: 500px;
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    padding: 15px;
    z-index: 1000;
    text-align: center;
    min-width: 400px;
    max-width: 800px !important; 
}

#context-options .form-check {
    display: inline-block;
    margin: 0 10px;
}

#context-options .form-check-label {
    font-weight: bold;
    color: #333;
}

#context-options textarea {
    resize: none;
    transition: height 0.2s ease-in-out;
    width: 100%;
    margin-top: 10px;
}
.close-button {
            cursor: pointer;
            background-color:white;
            color: black;
            border: none;
            padding: 5px 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
    </style>

<script>
 document.addEventListener("DOMContentLoaded", function() {
    const sendButton = document.getElementById("send-button");
    const chatInput = document.getElementById("chat-input");
    const chatBox = document.getElementById("chat-box");
    const contextInput = document.getElementById("context-input");
    const contextOptionYes = document.getElementById("contextYes");

    sendButton.addEventListener("click", function() {
        sendMessage();
    });

    chatInput.addEventListener("keypress", function(e) {
        if (e.key === "Enter") {
            sendMessage();
        }
    });

    document.querySelectorAll('input[name="contextOption"]').forEach((elem) => {
        elem.addEventListener("change", function(event) {
            if (event.target.value === "yes") {
                contextInput.classList.remove("d-none");
                contextInput.classList.add("d-block");
            } else {
                contextInput.classList.remove("d-block");
                contextInput.classList.add("d-none");
            }
        });
    });

    function sendMessage() {
        let message = chatInput.value.trim();
        if (message !== "") {
            let postData = {
                question: message,
                context: contextInput.value.trim()
            };
            if(!contextOptionYes.checked){
              postData['context'] = ""
            }
            // Make POST request to Flask API endpoint
            fetch('http://127.0.0.1:5000/answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(postData)
            })
            .then(response => response.json())
            .then(data => {
                // Append user's message
                appendMessage(message, "user");

                // Append context if provided
                if (contextInput.value.trim() !== "") {
                    appendMessage(`السياق: ${contextInput.value.trim()}`, "context");
                    contextInput.value = "";
                }

                if (data.error) {
                    appendMessage(`خطأ: ${data.error}`, "error");
                } else {
                    appendMessage(data.answer, "chatgpt");
                }
            })
            .catch(error => {
                console.error('Error:', error);
                appendMessage('حدث خطأ أثناء الاتصال بالخادم', "error");
            });

            // Clear input fields
            chatInput.value = "";
            contextInput.value = "";
        }
    }
    document.getElementById('context-toggle').addEventListener('click', function() {
            var contextOptions = document.getElementById('context-options');
            if (contextOptions.classList.contains('d-none')) {
                contextOptions.classList.remove('d-none');
                contextOptions.classList.add('d-block');
            } else {
                contextOptions.classList.remove('d-block');
                contextOptions.classList.add('d-none');
            }
        });

        document.getElementById('contextYes').addEventListener('click', function() {
            var contextInput = document.getElementById('context-input');
            contextInput.classList.remove('d-none');
            contextInput.classList.add('d-block');
        });

        document.getElementById('contextNo').addEventListener('click', function() {
            var contextInput = document.getElementById('context-input');
            contextInput.classList.remove('d-block');
            contextInput.classList.add('d-none');
        });
        document.getElementById('close-button').addEventListener('click', function() {
            var contextOptions = document.getElementById('context-options');
            contextOptions.classList.remove('d-block');
            contextOptions.classList.add('d-none');
        });

    function appendMessage(message, sender) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("chat-message");
        if (sender === "user") {
            messageElement.classList.add("user");
        } else if (sender === "context") {
            messageElement.classList.add("context");
        } else if (sender === "chatgpt") {
            messageElement.classList.add("chatgpt");
        } else if (sender === "error") {
            messageElement.classList.add("error");
        }

        const messageText = document.createElement("p");
        messageText.textContent = message;

        messageElement.appendChild(messageText);
        chatBox.appendChild(messageElement);

        // Scroll to the bottom of the chat box
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});

</script>
</body>
</html>