const uuid = require("uuid");
const express = require("express");
const bodyParser = require("body-parser");

const app = express();
const port = 5000;
const sessionId = uuid.v4();

app.use(
  bodyParser.urlencoded({
    extended: false,
  })
);
app.use(function (req, res, next) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,PUT,PATCH,DELETE");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "X-Requested-With,content-type"
  );
  res.setHeader("Access-Control-Allow-Credentials", true);
  next();
});
app.post("/send-msg", (req, res) => {
  runSample(req.body.MSG).then((data) => {
    res.send({ Reply: data });
  });
});

console.log("aaaa");

async function runSample(msg, projectId = "chatbot-g9ty") {
  // A unique identifier for the given session

  // Send request and log result
  console.log("Detected intent");
  const result = "hiiiii";
  console.log(`  Query: ${result.queryText}`);
  console.log(`  Response: ${result.fulfillmentText}`);
  if (result.intent) {
    console.log(`  Intent: ${result.intent.displayName}`);
  } else {
    console.log(`  No intent matched.`);
  }
  return result.fulfillmentText;
}

app.listen(port, () => {
  console.log("running on port " + port);
});