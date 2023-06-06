import wink from "wink-nlp";
import model from "wink-eng-lite-web-model";

const Classifier = require("wink-naive-bayes-text-classifier");
const bayesModel = require("./model.json");

const nlp = wink(model);
const its = nlp.its;

export const nbc = new Classifier();

const prepTask = (text) => {
  const tokens = [];
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
