

Okay, here is the plan of everything we want to do to deliver the essentially phase II of the entire research project. It starts by essentially having the classifiers run correctly on the golden set. What this means is that we need to have the golden set created and we need to have classifiers created and run on the golden set. The golden set before it needs to be annotated by a human being to have a golden standard. For that, the process roughly follows this sequence:
We need to get the files
Then we need to make sure that the files are here correctly, the golden set files are already done
Then we need to put the files into Markdown and we need to do a lot of quality assurance to make sure that they are correctly put into Markdown
There are multiple steps here like parsing them using regregs and then we are using Gemma to also do some additional corrections
We can also do another quality assurance here, both manual and automatic
After that is done, we can create tooling to manually annotate it to create a database which is of manual annotations/manual classifications
For everything that's going to be done later
Once we have the classification, once we have the database improper state, and we have the tooling that allows us to manually annotate, we can build the classifiers to manually annotate. The LLM classifiers are going to follow the exact same process that the human being is going to follow, and both should be saved separately, so there should be the human standard (human gold standard) and there should be the LLM run.

The human gold standard is going to be only one, and an LLM run is going to be per run. We need to save the IDs so we can benchmark them against human performance, so the golden standard. But for every single report, we're going to have chunks. When the reports are done, they're going to be chunked for every single AI mentioned per report. That's going to be in chunks.json, and it contains every single thing. Every single chunk can be easily annotated.
So essentially, all of that until we classify is preprocessing. Even the chunking is preprocessing. And then, everything that comes after chunking is what we call processing, which is classification. But we call this preprocessing and processing. The taxonomy of the classification can be found below: 




Adoption classifier: Only classify what kind of AI (in 3 categories) is being mentioned. 
De-prioritize the label of internal vs customer facing labels. 
Vendor extraction: (can de-prioritise this) but would be very interested in the findings. Just to see how much and who is being mentioned. 


We need to make changes to the golden set initial sample. The processing should look as follows:
We preprocess the documents from iXBRL/PDF to markdown that human readable. (We can skip this for now as a human being will be working with the iXBRL files directly to annotate them using a widget in the golden_set_annotation.ipynb 
Each preprocessed report to markdown is split into sections that contain a given keyword such as “AI”, “ML”, “LLM”, “Artificial Intelligence” etc. This chunking is going to be saved as the last step of pre-processing and first step of processing/classification. We should save in the database a metadata as to how many mentions and thus chunks were detected/created per document, so that we can track the AI mention propensity. That way we can also reliably track no mentions of AI. The only issue we might run into here, though perhaps we can leave it be for now, is the issue of multiple mentions per paragraph, for example a paragraph might contain multiple mentions of “AI” and one mention of “LLM” and one mention of “Artificial Intelligence”. As a result we would create multiple chunks based on each one of these mentions and thus classify them multiple times later on. This perhaps is a feature not a bug as measuring an AI-mention-dense paragraph multiple times can yield higher accuracy per classification and give us a more accurate overview of the company’s attitude towards AI. (For human annotators this is also not necessary as they are simply going to use Cmd + F for the same keywords using the browser when viewing the document). 
Now in the processing/classification stage: we are going to take each mention of AI and assign it a mention type(s). (It is important to note here that these mention types are not exclusive, and assigning all, or none, to a mention is fully permissible. For each tag that is assigned to a mention the AI judge should also assign it a level of confidence on this classification. We should provide the AI with examples of what plausible mentions might look like and what the tags (and their corresponding confidence) should look like) Mention types can be the following: 
Adoption (Any mention describing the deployment or the process of deployment of an AI system either within the company or for the client of the business) Any mention of this type with a not-low-level of confidence will be passed onto a next specialized classifier to classify the type of AI adoption mentioned (and assign its own confidence of that classification)
Risk (Any and all mentions of AI in the context of risk to the company or risk in general)
Harm (Any mention of a harm to a business as a result of AI or AI usage i.e. AI enabled misinformation or hacking). 
Vendor (Any AI mention that additionally contains a mention of the technology's vendor i.e. Microsoft or Google Cloud). Any mention of this type with a not-low-level of confidence will be passed onto a next specialized classifier to classify the type of vendor mentioned (and assign its own confidence of that classification)
General or Ambiguous (Anything that talks about future plans or AI as an opportunity without mentioning anything that would permit us to label it as a statement about AI causing harm, being tangibly used by the company or posing a substantive risk to the company) 
These tags should be processed as follows:
If the Adoption tag was assigned: Run this mention via the adoption_classifier, which given the text with the mention should classify it as either of the 3 categories: non-LLM, LLM, or Agentic AI (this split is quite simple but we should likely explain it in more detail to the LLM classifier)). There is no category for an ambiguous or unclear type of AI mentioned, instead when there isn’t enough information in the text to determine the type of AI being referred to, simply assign lower, or extremely low confidence levels. For example the following text: “We are deploying AI internally” can be classified as type: non-LLM, confidence: 0.1, type: LLM, confidence: 0.05. While a text as follows: “We have deployed our LLM chatbot to more than half of our employees” can be classified as type: type: LLM, confidence 0.95, type: non-LLM, confidence: 0.1, type: Agentic AI, confidence: 0.05.  
If the Risk tag was assigned : We want to pass it to a risk classifier that, similarly to the general mention_type classifier, will pick from a list of categories/tags to assign to the risk based on the defined taxonomy (the taxonomy can be found in the appendix) and give each assigned tag, if any, a confidence level. We want to give examples of the mentions and their ideal tags. We also want to add one variable here, which is the substantiveness metric - a score from 0 to 1 as to how substantive (1) or how much of a boilerplate (0) is the mention. 
If the Harm tag was assigned: We want to simply save this excerpt of text (or a reference to it) with the harm tag as well as the tag's confidence level. 
If the Vendor tag was assigned: We want to pass this mention to a Vendor classifier that has a short list of vendors (i.e. Google, Microsoft, OpenAI, internal, undisclosed, other(specify) ). Each tag should also not be exclusive and have its own assigned confidence level by the LLM. 
If the General or Ambiguous tag was assigned: We want to simply save this excerpt of text (or a reference to it) with the general_or_ambiguous tag as well as the tags confidence level. 


When passing the text to classifiers, iInitially we might want to pass all levels of confidence (confidence level >0) but this should be a variable that we can adjust easily later, as we might find that mentions below confidence of 0.2 are not adequate.  

Please make sure that the prompts used by the LLM classifiers are in a separate yaml file for easy access and iterative improvements. 

Appendix:
The taxonomy for AI risk classification:

Name
Description
Operational & Technical Risk
Model failures, bias, reliability, system errors
Cybersecurity Risk
AI-enabled attacks, data breaches, system vulnerabilities
Workforce Impacts
Job displacement, skill requirements, automation
Regulatory & Compliance Risk
Legal liability, compliance costs, AI regulations
Information Integrity
Misinformation, content authenticity, deepfakes
Reputational & Ethical Risk
Public trust, ethical concerns, human rights, bias
Third-Party & Supply Chain Risk
Vendor reliance, downstream misuse, LLM provider dependence
Environmental Impact
Energy use, carbon footprint, sustainability
National Security Risk
Geopolitical, export controls, adversarial use



We might want to add “Failure to Adopt AI”, as that seems to be mentioned a lot. An experiment we could run afterwards is to let the model create its own tags of risk type, to see what types of AI as a risk mentions are the most prevalent. This is a future and optional experiment. 



Appendix: List of companies to process
ID
Company
Sector
Index
Type
1
Croda International plc
Chemicals
FTSE 100
Best proxy
2
Rolls-Royce Holdings plc
Civil Nuclear/Space
FTSE 100
Best proxy
3
BT Group plc
Communications
FTSE 100
Direct
4
BAE Systems plc
Defence
FTSE 100
Direct
5
Serco Group plc
Government Services
FTSE 250
Best proxy
6
Shell plc
Energy (Extraction)
FTSE 100
Direct
7
Lloyds Banking Group plc
Finance (Banking)
FTSE 100
Direct
8
Tesco plc
Food (Retail)
FTSE 100
Direct
9
AstraZeneca plc
Health (Pharma)
FTSE 100
Direct
10
National Grid plc
Energy (Transmission)
FTSE 100
Direct
11
Severn Trent plc
Water
FTSE 100
Direct
12
Aviva plc
Insurance
FTSE 100
Direct
13
Schroders plc
Asset Management
FTSE 100
Direct
14
FirstGroup plc
Transport
FTSE 250
Direct
15
Clarkson plc
Shipping
FTSE 250
Direct

Appendix: Data Layout
/data/raw/ for original iXBRL files and download metadata.
/data/processed/<run_id>/documents/ for cleaned, readable markdown outputs.
/data/processed/<run_id>/metadata/ for JSON sidecars per document.
/data/processed/<run_id>/documents_manifest.json for processed document index.
/data/processed/<run_id>/chunks/ for AI-mention chunks (JSONL).
/data/results/ reserved for classifier outputs later.
