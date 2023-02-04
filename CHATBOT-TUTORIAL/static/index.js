var $messages = $(".messages-content");
var serverResponse = "wala";
var informations = 0;
var suggession;
var label = document.createElement("label");
label.id = "label_information";
document.getElementsByTagName("body")[0].appendChild(label);
var yesOrno = "";

//speech reco
try {
  var SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  var recognition = new SpeechRecognition();
} catch (e) {
  console.error(e);
  $(".no-browser-support").show();
}

$("#start-record-btn").on("click", function (e) {
  recognition.start();
});

recognition.onresult = (event) => {
  const speechToText = event.results[0][0].transcript;
  document.getElementById("MSG").value = speechToText;
  //console.log(speechToText)
  insertMessage();
};

function listendom(no) {
  console.log(no);
  //console.log(document.getElementById(no))
  document.getElementById("MSG").value = no.innerHTML;
  insertMessage();
}

$(window).load(function () {
  $messages.mCustomScrollbar();
  setTimeout(function () {
    serverMessage(
      "hello i am customer support bot type hi and i will show you quick buttions"
    );
  }, 100);
});

function updateScrollbar() {
  $messages.mCustomScrollbar("update").mCustomScrollbar("scrollTo", "bottom", {
    scrollInertia: 10,
    timeout: 0,
  });
}

function insertMessage() {
  msg = $(".message-input").val();
  if ($.trim(msg) == "") {
    return false;
  }
  $('<div class="message message-personal">' + msg + "</div>")
    .appendTo($(".mCSB_container"))
    .addClass("new");
  fetchmsg();

  $(".message-input").val(null);
  updateScrollbar();
}

document.getElementById("mymsg").onsubmit = (e) => {
  e.preventDefault();
  insertMessage();
};

function serverMessage(response2) {
  if ($(".message-input").val() != "") {
    return false;
  }
  $(
    '<div class="message loading new"><figure class="avatar"><img src="/static/bot.png" /></figure><span></span></div>'
  ).appendTo($(".mCSB_container"));
  updateScrollbar();

  setTimeout(function () {
    $(".message.loading").remove();
    $(
      '<div class="message new"><figure class="avatar"><img src="/static/bot.png" /></figure>' +
        response2 +
        "</div>"
    )
      .appendTo($(".mCSB_container"))
      .addClass("new");
    updateScrollbar();
  }, 100 + Math.random() * 20 * 100);
}

function fetchmsg() {
  var url = "http://127.0.0.1:5000/get-keyword?prompt=";

  const data = new URLSearchParams();
  for (const pair of new FormData(document.getElementById("mymsg"))) {
    data.append(pair[0], pair[1]);
    yesOrno = pair[1];
  }
  url+=data.get('MSG')
  fetch(url,{ 
  headers: {
  'Content-Type': 'text/plain',
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE",
  "Access-Control-Allow-Headers":
  "X-Requested-With,content-type",
  "Access-Control-Allow-Credentials": true
  },
})  
.then(r => r.json()).then(data=> {
  // `data` is the parsed version of the JSON returned from the above endpoint.
  console.log(data);  // { "userId": 1, "id": 1, "title": "...", "body": "..." }
        // console.log(response)
      var res = data.result;
      // if (informations < 12) {
      //   check(informations, response.Reply);
      //   if (
      //     response.Reply ===
      //     "Ok, i will help you to buy house i will ask you a few questions and answer that first in wich city you like to buy a house?"
      //   ) {
      //     informations++;
      //   }

        serverMessage(res);
      // } else {
        // res = "Ok thx you for working for us now click the button (שליחה)";
      //   serverMessage(res);
      // }
      speechSynthesis.speak(new SpeechSynthesisUtterance(res));
    })
    .catch((error) => console.error("Error h:", error));
}
function check(index, str) {
  if (informations >= 1) {
    all = str.split(" ");
    result = "";
    console.log(str.split(" "));
    if (informations === 1) {
      for (i = 0; i < all.length; i++) {
        if (all[i] === "now") break;
        if (i >= 6) {
          result += all[i] + " ";
        }
      }
    }
    if (informations === 2) {
      for (i = 0; i < all.length; i++) {
        if (all[i] === "now") break;
        if (i >= 6) {
          result += all[i] + " ";
        }
      }
    }
    if (informations === 3) {
      for (i = 0; i < all.length; i++) {
        if (all[i] === "now") break;
        if (i >= 5) {
          result += all[i] + " ";
        }
      }
    }
    if (informations === 4) {
      for (i = 0; i < all.length; i++) {
        if (all[6] === "to") {
          if (all[i] === "now") break;
          if (i >= 5 && i !== 6) {
            result += all[i] + " ";
          }
        } else {
          if (all[i] === "now") break;
          if (i >= 6) {
            result += all[i] + " ";
          }
        }
      }
    }
    if (informations === 5) {
      for (i = 0; i < all.length; i++) {
        if (all[i] === "now") break;
        if (all[4] === "rooms") {
          if (i >= 6) {
            result += all[i] + " ";
          }
        } else {
          result = all[4] + " ";
          break;
        }
      }
    }
    if (informations === 6) {
      for (i = 0; i < all.length; i++) {
        if (all[i] === "now") break;
        if (i >= 4 && i !== 5) {
          result += all[i] + " ";
        }
      }
    }
    if (informations === 7) {
      for (i = 0; i < all.length; i++) {
        if (all[i] === "now") break;
        if (i >= 7) {
          if (all[i] !== "and") result += all[i] + " ";
        }
      }
    }
    if (informations >= 8) {
      if (yesOrno === "yes") result = "true";
      else result = "false";
    }
    console.log(informations);
    result += ";";
    label.innerText = label.innerText + result;
    console.log(label.innerText);
    informations++;
  }
}