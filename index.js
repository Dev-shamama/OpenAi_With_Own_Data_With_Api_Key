import { OpenAI } from "langchain/llms/openai";
import { RetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// Loaders
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

import { config } from "dotenv";
import { Document } from "langchain/document";

config();

const loader = new DirectoryLoader("./docs", {
  ".json": (path) => new JSONLoader(path),
  ".txt": (path) => new TextLoader(path),
  ".csv": (path) => new CSVLoader(path),
  ".pdf": (path) => new PDFLoader(path),
});

const docs = await loader.load();

const csvContent = docs.map((doc) => doc.pageContent);

const askModel = async (question) => {
  const model = new OpenAI({ openAIApiKey: process.env.OPENAI_API_KEY });

  let vectorstore;

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 900,
  });

  const splitDoc = await textSplitter.createDocuments(csvContent);

  vectorstore = await HNSWLib.fromDocuments(
    splitDoc,
    new OpenAIEmbeddings()
  )

  await vectorstore.save("MyVector.index");
  console.log("Vector store is created")

  const chain = RetrievalQAChain.fromLLM(model, vectorstore.asRetriever())
  console.log("Querying...")

  const res = await chain.call({query: question})
    console.log(res);
};

askModel("summary fetch api with code");