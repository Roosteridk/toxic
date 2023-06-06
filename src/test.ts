import wink from "wink-nlp";
import model from "wink-eng-lite-web-model";
import * as fs from "node:fs";
import { parse } from "csv-parse/sync";

const Classifier = require("wink-naive-bayes-text-classifier");

const nlp = wink(model);
const its = nlp.its;

const testData = fs.readFileSync("test.csv");
const testLabels = fs.readFileSync("test_labels.csv");

const testRecords = parse(testData, { columns: true }) as {
  id: string;
  comment_text: string;
}[];
const testLabelsRecords = parse(testLabels, { columns: true }) as {
  id: string;
  toxic: string;
  severe_toxic: string;
  obscene: string;
  threat: string;
  insult: string;
  identity_hate: string;
}[];

console.log(testRecords.length, testLabelsRecords.length);

const test = testRecords.map((record) => {
  const label = testLabelsRecords.find((r) => r.id === record.id)!;
  const comment = record.comment_text;
  // Skip records with -1
  if (
    label.toxic === "-1" || label.severe_toxic === "-1" ||
    label.obscene === "-1" || label.threat === "-1" ||
    label.insult === "-1" || label.identity_hate === "-1"
  ) {
    return null;
  }

  // Label is 1 if any of the labels are 1
  const isToxic = label.toxic === "1" || label.severe_toxic === "1" ||
    label.obscene === "1" || label.threat === "1" ||
    label.insult === "1" || label.identity_hate === "1";

  return { id: record.id, comment, label: isToxic ? 1 : 0 };
});

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

const nbc = new Classifier();
nbc.definePrepTasks([prepTask]);

const json = fs.readFileSync("./model.json", "utf8");
nbc.importJSON(json);
nbc.consolidate();

for (let i = 0; i < test.length; i++) {
  const record = test[i];
  if (!record) {
    continue;
  }
  const { id, comment, label } = record;
  nbc.evaluate(comment, label);
  console.log(`Progress: ${i}/${test.length}`);
}

console.log(nbc.metrics());
