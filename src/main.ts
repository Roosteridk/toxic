import {
  ConfigFormBuilder,
  Context,
  Devvit,
  getSettings,
} from "@devvit/public-api";
import wink from "wink-nlp";
import model from "wink-eng-lite-web-model";
import bayesModel from "./model.js";

export default Devvit;

// Initialize the API endpoints
const api = Devvit.use(Devvit.Types.RedditAPI.LinksAndComments);
const mod = Devvit.use(Devvit.Types.RedditAPI.Moderation);

// We use a Naive Bayes classifier to determine the probability that a comment is toxic
// The classifier was trained on the dataset from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
// In the future, we may train our own classifier on a different dataset and with a different model
const Classifier = require("wink-naive-bayes-text-classifier");

const nlp = wink(model);
const its = nlp.its;
const nbc = new Classifier();
const prepTask = (text: string) => {
  const tokens: string[] = [];
  nlp.readDoc(text)
    .tokens()
    .filter((t) => (t.out(its.type) === "word" && !t.out(its.stopWordFlag)))
    .each((t) =>
      tokens.push(
        (t.out(its.negationFlag)) ? "!" + t.out(its.stem) : t.out(its.stem),
      )
    );
  return tokens;
};
nbc.definePrepTasks([prepTask]);
nbc.importJSON(bayesModel);
nbc.consolidate();

Devvit.addAction({
  name: "Tox-Screen",
  description: "Analyzes a comment for toxicity",
  context: Context.COMMENT,
  handler: async (event) => {
    const body = event.comment.body!;
    const label = nbc.predict(body);
    console.log(nbc.computeOdds(body));
    return {
      success: true,
      message: `☣️ The comment ${
        label == 1 ? "might be toxic." : "is probably not toxic."
      }`,
    };
  },
});

Devvit.addAction({
  name: "Tox-Label",
  description: "Label a comment for toxicity or lack thereof",
  context: Context.COMMENT,
  userInput: new ConfigFormBuilder().booleanField(
    "userLabel",
    "Check if you think the comment is toxic",
  ).build(),
  handler: async (event) => {
    const body = event.comment.body!;
    const label = nbc.predict(body);
    // Calculate SHA-256 hash of the comment body
    const hash = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(body),
    );
    // Get first 4 bytes of hash as a hex string
    const hashStr = Array.from(new Uint8Array(hash));
    const userLabel = event.userInput!.fields[0].response;
    const report = {
      thingId: event.comment.id!,
      reason: "Toxicity - " + (label == 1 ? "Likely" : "Unlikely") + " | " +
        "User Label: " + (userLabel ? "Toxic" : "Not Toxic") + " | " +
        "Hash: " + hashStr.slice(0, 4).map((b) => b.toString(16)).join(""),
    };
    console.log(report);
    await api.Report(report);
    return {
      success: true,
      message: "☣️ The comment was reported for toxicity.",
    };
  },
});

Devvit.addSettings([{
  type: "boolean",
  name: "remove-flagged",
  label:
    "Remove flagged comments. Note: This is dangerous as it may result in the deletion of false positives",
}]);

// Listens for comment submissions and flags them if they are toxic
Devvit.addTrigger({
  event: Devvit.Trigger.CommentSubmit,
  handler: async (event, meta) => {
    const body = event.comment!.body!;
    const label = nbc.predict(body);
    const settings = await getSettings(meta);
    const remove = settings["remove-flagged"] as boolean;
    if (label == 1) {
      if (remove) {
        await mod.Remove({
          id: event.comment!.id!,
          spam: false,
        });
      }
      await api.Report({
        thingId: event.comment!.id!,
        reason: "Toxicity",
      });
    }
  },
});
