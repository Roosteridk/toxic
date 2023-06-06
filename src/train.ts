import wink from "wink-nlp";
import model from "wink-eng-lite-web-model";
import * as fs from "node:fs";
import { parse } from "csv-parse/sync";

const Classifier = require("wink-naive-bayes-text-classifier");

const nlp = wink(model);
const its = nlp.its;

// "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
// Binarize the labels to 0 or 1
// Convert to JSON with the following format:
// {
//   "id": "0000997932d777bf",
//   "comment_text": "Explanation\nWhy the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms, just closure on some GAs after I voted at New York Dolls FAC. And please don't remove the template from the talk page since I'm retired now.
//   "label": 1 // 0 or 1 If any of the labels are 1, then label is 1
// }

export type Record = {
  id: string;
  comment_text: string;
  toxic: string;
  severe_toxic: string;
  obscene: string;
  threat: string;
  insult: string;
  identity_hate: string;
};

// Read the CSV file
const input = fs.readFileSync("train.csv");
const records = parse(input, { columns: true }) as Record[];
const data = records.map((record) => {
  const comment = record.comment_text;
  const label = record.toxic === "1" || record.severe_toxic === "1" ||
      record.obscene === "1" || record.threat === "1" ||
      record.insult === "1" || record.identity_hate === "1"
    ? 1
    : 0;
  return { id: record.id, comment, label };
});

const nbc = new Classifier();

// Preprocess the text
// 1. Tokenize
// 2. Remove stop words
// 3. Stem
// 4. Negate if negation flag is set
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
nbc.defineConfig({ considerOnlyPresence: true, smoothingFactor: 0.5 });

// Train the model

for (let i = 0; i < data.length; i++) {
  console.log(`Training ${i} of ${data.length}`);
  nbc.learn(data[i].comment, data[i].label);
}

nbc.consolidate();

fs.writeFileSync("model.json", nbc.exportJSON());
