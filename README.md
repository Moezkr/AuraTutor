<h1 align="center"> 🌌 AuraTutor: Gamified AI Educational Ecosystem 🌌 </h1>

<p align="center">
  Welcome to <b>AuraTutor</b>. This is not just an educational platform; it is a fully autonomous, gamified, AI-driven remediation ecosystem designed to revolutionize how students learn and how teachers monitor cognitive development. Powered by state-of-the-art Machine Learning (PyTorch, xLSTM) and real-time concept drift detection (ADWIN, KSWIN), AuraTutor dynamically adapts to the human brain in real-time.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-15-black?style=for-the-badge&logo=next.js" alt="Next.js" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
</p>

---

## 🌟 Table of Contents
1. [Core Ecosystem Overview](#core-ecosystem-overview)
2. [xGATES: The Learner's Dimension](#xgates-the-learners-dimension)
3. [xCARE: The Teacher's Command Center](#xcare-the-teachers-command-center)
4. [AI Concept Drift & Remediation Engine](#ai-concept-drift--remediation-engine)
5. [Flow Diagrams & Architectural Explanations](#flow-diagrams--architectural-explanations)
6. [Demo Pictures](#demo-pictures)


---

## 🚀 Core Ecosystem Overview

AuraTutor bridges the gap between high-dopamine gamification and rigorous academic pedagogy. By utilizing a dual-dashboard architecture, the platform isolates the student's learning experience from the teacher's analytical overhead, ensuring that both parties have access to tools specifically engineered for their needs.

The entire system is monitored by an autonomous AI agent that tracks **keystrokes, quiz answer delays, success rates, and historical cognitive patterns**. If the system detects a student is struggling, it initiates an "AI Remediation Protocol," generating dynamic hints, altering the difficulty of upcoming questions, and notifying the teacher of a potential "Concept Drift."

---

## 🎓 xGATES: The Learner's Dimension

The **xGATES Dashboard** is the student's portal. Designed with a dark, futuristic, "Matrix-like" aesthetic featuring neon green accents and smooth glassmorphism, it transforms learning into an epic digital quest.

### 1. The Quest Realms & Dynamic Quizzes
Instead of standard tests, students enter "Quest Realms." Each realm represents a curriculum module.
* **Adaptive Difficulty:** As the student answers correctly, the AI dynamically ramps up the difficulty. If they fail, the system temporarily reduces difficulty to rebuild confidence.
* **Instant AI Remediation:** When a student submits a wrong answer, the AI does not just give them the correct answer. It generates a Socratic hint—a custom-tailored explanation that guides the student to the correct logic based on the specific mistake they made.
* **Interactive UI:** Quizzes feature animated transitions, drag-and-drop elements, and real-time validation to keep cognitive engagement high.

### 2. Gamification: Leaderboards, XP, & Clan Wars
Engagement is driven by a massive, platform-wide gamification engine:
* **Global Leaderboard:** Students earn Experience Points (XP) for completing quizzes, maintaining streaks, and participating in platform events. The leaderboard ranks students globally and within their specific classes.
* **Badges & Achievements:** Students unlock high-quality digital badges for milestones (e.g., "Flawless Victory" for a 100% quiz score, "Night Owl" for studying past midnight).
* **Clan Wars:** Students can join "Clans" (study groups). Clan scores are aggregated, and clans compete in weekly educational wars where solving collaborative AI-generated puzzles earns massive XP multipliers.
* **Daily Streaks:** A visual fire-streak counter motivates students to log in and complete at least one micro-lesson every single day.

### 3. Personal Analytics & Skill Trees
* Students can view their own "Skill Tree," visualizing their mastery over different subjects.
* Weak points are highlighted in red, prompting the student to initiate a "Remediation Quest" to turn that skill green.

---

## 👑 xCARE: The Teacher's Command Center

The **xCARE Dashboard** is built for educators, administrators, and curriculum designers. It provides an omniscient view of the entire student body through advanced telemetry and AI diagnostics.

### 1. Advanced Real-Time Telemetry
* **Live Heatmaps:** Teachers can view a live heatmap of cognitive engagement. They can see exactly which questions in a quiz are causing the highest failure rates across an entire cohort.
* **Student Risk Profiling:** The AI automatically flags students who are at risk of failing based on historical data trends, allowing teachers to intervene before a final exam.
* **Engagement Metrics:** Track how much time students spend reading materials versus guessing on quizzes. The system flags "spam guessing" behaviors.

### 2. Full Curriculum Management (CRUD)
* **Quiz Builder:** A drag-and-drop interface for creating complex quizzes. Teachers can manually input questions or prompt the AI to auto-generate a 50-question quiz based on a specific syllabus topic.
* **Material Distribution:** Teachers can assign specific modules, PDFs, and video links to targeted groups of students or entire classes.

### 3. AI-Assisted Grading & Feedback
* For open-ended questions, the AI performs a preliminary grading pass, highlighting keywords and suggesting a score. The teacher has the final say but saves hours of manual grading time.
* The system aggregates common misconceptions and generates a "Teacher Report" summarizing the class's overall understanding of a concept.

---

## 🧠 AI Concept Drift & Remediation Engine: A Deep Dive

The true power of AuraTutor lies in its backend AI, an autonomous orchestration engine that actively monitors the stability of student learning patterns. This section provides an exhaustive, academic-level breakdown of the mathematical models, algorithms, and pedagogical frameworks that drive our system.

### Understanding Educational Concept Drift
In the context of Educational Data Mining (EDM) and continuous machine learning, "Concept Drift" refers to a fundamental shift in the statistical properties of the target variable over time. In layman's terms: it is the exact mathematical moment when a student's cognitive model diverges from the system's expected predictive model.

For example, if a student consistently scores 90% across foundational modules, the system builds an internal confidence matrix predicting continued success. However, when a complex new topic (e.g., "Advanced Neural Networks") is introduced, and the student's performance drops to 40%, the underlying statistical distribution of their cognitive telemetry has "drifted." This is not merely a failed quiz; it is a structural anomaly indicating a failure in knowledge transfer.

AuraTutor classifies Concept Drift into four distinct archetypes:
1. **Sudden Drift (Abrupt Shift):** The student hits a complete conceptual wall. Performance drops instantly from high to low. This typically indicates a missing prerequisite skill.
2. **Gradual Drift:** Performance oscillates between high and low, slowly trending downward. This indicates cognitive fatigue or a partial misunderstanding of a core concept that is increasingly compounding.
3. **Incremental Drift:** A slow, continuous degradation of performance over a long period. This often signals memory decay (forgetting previously learned concepts).
4. **Recurrent Drift:** Performance drops periodically in specific, repeating contexts (e.g., struggling only when mathematical formulas are presented, but excelling in theoretical questions).

### The Algorithmic Foundation
To detect these drifts in real-time, our PyTorch-powered backend utilizes state-of-the-art sliding window algorithms and non-parametric statistical tests. The system continuously ingests a high-velocity stream of telemetry data (quiz scores, response latency, hint request frequency).

#### 1. ADWIN (Adaptive Windowing)
ADWIN is a parameter-free algorithm designed to detect drift in a data stream by automatically adjusting the size of a sliding window. 
* **Mechanism:** ADWIN maintains a window `W` of the most recent student performance metrics. It continuously checks if two large enough sub-windows of `W` exhibit distinct averages. 
* **Mathematical Trigger:** Let `W` be split into `W0` (older data) and `W1` (newer data). If the absolute difference between the expected value of `W0` and `W1` exceeds a dynamically calculated threshold (epsilon), ADWIN strictly concludes that the underlying distribution has changed.
* **Advantage in AuraTutor:** Because ADWIN automatically grows the window when the student's performance is stable and shrinks it when a change occurs, it prevents "false alarms" caused by a single accidental wrong answer, while remaining hyper-sensitive to true cognitive failure.

#### 2. KSWIN (Kolmogorov-Smirnov Windowing)
KSWIN is based on the Kolmogorov-Smirnov (KS) statistical test, which is entirely non-parametric, meaning it makes no assumptions about the underlying distribution of the student's data.
* **Mechanism:** KSWIN maintains a fixed-size memory of recent events (e.g., the last 100 interactions). It divides this memory into a historical baseline window and a recent comparative window.
* **Mathematical Trigger:** The KS test evaluates the maximum distance between the empirical cumulative distribution functions (eCDF) of the two windows. If this distance (the KS statistic) is significantly large, and the resulting p-value falls below a critical threshold (alpha), KSWIN formally rejects the null hypothesis that the data comes from the same distribution.
* **Advantage in AuraTutor:** KSWIN is exceptionally powerful for detecting sudden cognitive drops. It excels in recognizing when a student's interaction latency (time taken to answer) suddenly spikes, even if their final answer was correct, indicating hesitation and a lack of fluency.

#### 3. Page-Hinkley Test (PHT)
AuraTutor also employs the Page-Hinkley test as a secondary validation layer for detecting incremental drifts (slow memory decay).
* **Mechanism:** PHT tracks the cumulative difference between the observed student performance and their historical mean. 
* **Advantage in AuraTutor:** By utilizing a cumulative sum, PHT is highly resistant to noise. It allows the system to identify students who are slowly burning out over the course of a semester before they reach a critical failure point.

### The Autonomous Cognitive Remediation Protocol
Detecting the drift is only the first half of the architecture. The true innovation of AuraTutor is its autonomous response mechanism. When an algorithm (ADWIN, KSWIN, or PHT) fires a Drift Signal, the system freezes standard progression and initiates a highly orchestrated "Remediation Protocol."

#### Phase 1: Micro-Diagnostic Rerouting
Immediately upon drift detection, the student is isolated from the main curriculum. The system generates a "micro-diagnostic" quiz—a sequence of 3 to 5 highly specific questions designed not to test the current complex topic, but to test the *prerequisite* nodes of that topic.
* *Example:* If a student fails a Calculus question on "Derivatives of Trigonometric Functions," the micro-diagnostic will quietly test their basic Algebra and Unit Circle knowledge to find the exact broken link in their knowledge chain.

#### Phase 2: Dynamic Socratic Intervention
Once the specific prerequisite failure is identified, the PyTorch LLM engine generates a dynamic intervention.
* The system does not provide the direct answer. Instead, it utilizes Socratic questioning, generating hints that guide the student to realize their own logical fallacy.
* These hints are injected directly into the xGATES interface using localized, animated "AI Tutors" to reduce the stigma of failure and increase gamified engagement.

#### Phase 3: The Adaptive Recovery Window
While the student is navigating the remediation path, the Drift Algorithms initialize a "Recovery Window."
* The algorithms now monitor the student's telemetry against the remediation content.
* If the student successfully completes the remediation, the model parameters are updated to reflect the repaired knowledge node, and the student is seamlessly merged back into the main Quest Realm.
* If the student fails the remediation, the drift is escalated to a "Critical Failure."

### Phase 4: eXplainable AI (XAI) and Teacher Telemetry
AuraTutor operates as an "Explainable AI" (XAI) system. It is unacceptable for an educational AI to simply state "Student is failing." It must explain *why*. When a drift is detected and escalated, the xCARE (Teacher) Dashboard receives a comprehensive diagnostic report generated by the Explainability Module.

This report is broken down into three distinct taxonomies:
1. **Temporal Explanation (The "When"):**
   * The exact timestamp and contextual trigger of the drift.
   * Example: "Drift initiated on Tuesday at 14:02 during Module 4. The student's response latency increased by 400% immediately following the introduction of the 'Backpropagation' concept."
2. **Structural Explanation (The "What"):**
   * An analysis of the specific algorithms triggered.
   * Example: "KSWIN detected a sudden drift in confidence intervals. The internal model structure has temporarily decreased the student's mastery coefficient for 'Calculus Prerequisites' from 0.85 to 0.42."
3. **Semantic Explanation (The "Why"):**
   * A human-readable synthesis generated by the LLM detailing the root pedagogical cause.
   * Example: "The student has conceptually confused the Chain Rule with the Product Rule. The AI Remediation Protocol has been activated to review foundational polynomial derivatives."

### Pedagogical Impact and Scalability
By automating the detection of cognitive failure and immediately initiating targeted remediation, AuraTutor solves the "Two Sigma Problem" of education—providing the benefits of 1-on-1 personalized tutoring at a massive, automated scale.

The architecture is highly scalable. Because the drift algorithms operate on efficient sliding windows and memory-bounded matrices, they can process millions of student interactions per second with sub-millisecond latency. Whether managing a single classroom of 30 students or a global MOOC with 300,000 active learners, the core engine guarantees that no student falls victim to unseen cognitive drift.

---


## 🌊 Flow Diagrams & Architectural Explanations

Below are the detailed flowcharts that dictate the logic of the AuraTutor intelligent system.

### Diagram 1: Continuous Monitoring & Drift Adaptation Flow


<p align="center">
  <img width="805" height="1049" alt="Workflow complet" src="https://github.com/user-attachments/assets/85ba8e89-eb0c-44fc-9ba4-e0f70cdb8093" />
</p>

**Deep-Dive Description:**
This flow represents the continuous, non-stop heartbeat of the educational AI.
1. **Acquisition des données en flux (Data Stream):** The system continuously ingests telemetry (clicks, time spent, quiz answers) representing the current environment.
2. **Prédiction par le modèle courant:** The active Continuous Learning model produces predictions regarding the student's expected performance.
3. **Surveillance continue (Monitoring):** Constant tracking of performance indicators (precision, error rate, stability) and statistical distribution of input variables.
4. **Détection de la dérive conceptuelle:** Application of drift algorithms (ADWIN, DDM, Page-Hinkley). If a significant change occurs, a **Drift Signal** is fired.
5. **Analyse et caractérisation de la dérive:** The system categorizes the drift as sudden, gradual, incremental, or recurrent.
6. **Sélection de la stratégie d'adaptation:** Based on the characterization, a strategy is chosen (e.g., partial retraining, temporal weighting, memory mechanisms).
7. **Mise à jour du modèle (Adaptation):** The model adjusts its parameters or structure, integrating new data without catastrophic forgetting.
8. **Génération des explications (Explicabilité):** The system generates human-readable explanations covering:
   - *Temporal:* When the drift was detected.
   - *Structural:* What components of the model were modified.
   - *Semantic:* Which variables contributed most to the drift.
9. **Traçabilité et journalisation:** All events, decisions, and performance effects are permanently logged for audit and review.

### Diagram 2: Educational Drift Detection System (Mind Map)


<p align="center">
  <img width="3448" height="4537" alt="WorkFlow_sujet1" src="https://github.com/user-attachments/assets/2ad5d65e-3185-46f4-ad95-88d0acdabe7e" />
</p>

**Deep-Dive Description:**
This mind map outlines the hierarchical taxonomy of the entire detection platform:
* **1. Objectifs du système:** The primary goals: Predicting performance, detecting changes, explaining evolutions, and managing the delicate balance between comprehension and adaptation.
* **2. Collecte des données:** The raw materials: Grades, quizzes, time spent, UI interactions, and continuous temporal flows.
* **3. Prétraitement:** The cleaning phase: Normalization, noise reduction, and variable selection.
* **4. Apprentissage initial:** The baseline models: Decision Trees, Logistic Regression, Random Forests.
* **5-11. The Processing Loop:** Moving from Deployment/Prediction -> Surveillance -> Drift Detection -> Categorization -> Explanation (The Key Step) -> Visualization -> Continuous Looping.

### Diagram 3: Intelligent System Input/Output Architecture


<p align="center">
  <img width="1536" height="1024" alt="Système d_apprentissage continu adaptatif" src="https://github.com/user-attachments/assets/b15c361e-e4ee-4680-8f74-d3228eae182e" />
</p>

**Deep-Dive Description:**
This is the high-level infrastructure view of the AI microservice.
* **Input (Left):** A continuous, uninterrupted stream of student interaction data flows from the cloud into the Intelligent System, alongside data from the legacy model.
* **Système Intelligent (Center):** The core engine executes three simultaneous loops:
  1. *Détection de dérive* (Finding the anomaly)
  2. *Adaptation du modèle* (Fixing the anomaly)
  3. *Explicabilité* (Explaining the anomaly to the Teacher)
* **Output (Right):** The system outputs highly accurate, newly adapted predictions (*Prédictions adaptatives*) and generates comprehensible, human-readable explanations (*Explications compréhensibles*) that are rendered directly onto the Next.js frontend dashboards.

---

## 😄 Demo Pictures

### 🏠 Home Page UI

|<img width="1919" height="940" alt="Screenshot 2026-06-17 133301" src="https://github.com/user-attachments/assets/e5de6ad1-303f-4f23-b8bb-1b3b442abb55" />|<img width="1918" height="939" alt="Screenshot 2026-06-17 133326" src="https://github.com/user-attachments/assets/b853ea14-c831-4d0b-b58e-58aee79b3e8b" />|<img width="1919" height="939" alt="Screenshot 2026-06-17 133354" src="https://github.com/user-attachments/assets/05cf0a42-2c72-4a2b-924d-55af89cbc18a" />|
|---------|---------|---------|

### 👑 xCARE Dashboard

|<img width="1909" height="941" alt="Screenshot 2026-06-17 133414" src="https://github.com/user-attachments/assets/0d6b8343-91ed-4a4c-a84b-f496df91c93f" />|<img width="1909" height="938" alt="Screenshot 2026-06-17 133451" src="https://github.com/user-attachments/assets/64f9a94d-e104-4003-a55d-5c7b380432bf" />|<img width="1909" height="941" alt="Screenshot 2026-06-17 133504" src="https://github.com/user-attachments/assets/091cf7a0-e6e2-41d6-8f0b-be5477445dfe" />|
|---------|---------|---------|

|<img width="1909" height="941" alt="Screenshot 2026-06-17 133540" src="https://github.com/user-attachments/assets/8f20f907-b382-4596-9010-462bd4546d7b" />|<img width="1910" height="939" alt="Screenshot 2026-06-17 133603" src="https://github.com/user-attachments/assets/288a43cb-58e9-4052-b308-e009357fd139" />|<img width="1909" height="940" alt="Screenshot 2026-06-17 134100" src="https://github.com/user-attachments/assets/4f823d12-e7cd-43ac-8bdf-e0983ce2bc6e" />|
|---------|---------|---------|

|<img width="1910" height="936" alt="Screenshot 2026-06-17 134408" src="https://github.com/user-attachments/assets/9c3c8185-1f70-4e42-80dd-533a2b36a0cd" />|<img width="1908" height="935" alt="Screenshot 2026-06-17 134455" src="https://github.com/user-attachments/assets/d0101a94-024b-4f32-9fb1-46760c6a851f" />|<img width="1908" height="905" alt="Screenshot 2026-06-17 134507" src="https://github.com/user-attachments/assets/8b8c7233-485f-4580-b377-fd01982224c8" />|
|---------|---------|---------|

|<img width="1910" height="939" alt="Screenshot 2026-06-17 134649" src="https://github.com/user-attachments/assets/a32561ce-31b3-4303-a74a-48b8cdfdf330" />|<img width="1910" height="939" alt="Screenshot 2026-06-17 134721" src="https://github.com/user-attachments/assets/997f2452-cb76-44a2-8712-96681349cde8" />|<img width="1919" height="939" alt="Screenshot 2026-06-17 134732" src="https://github.com/user-attachments/assets/d5734e95-e06a-4ef3-8977-df9a4fb1a18b" />|
|---------|---------|---------|

|<img width="1911" height="945" alt="Screenshot 2026-06-17 134805" src="https://github.com/user-attachments/assets/962f603e-a2d9-4edd-9dfa-7e77856fd697" />|<img width="1910" height="938" alt="Screenshot 2026-06-17 134859" src="https://github.com/user-attachments/assets/2c900621-8aa2-4ec7-8544-760d8613ae60" />|<img width="1914" height="934" alt="Screenshot 2026-06-17 134922" src="https://github.com/user-attachments/assets/fb7173a6-a925-4676-8d95-a4085fea1fda" />|
|---------|---------|---------|

|<img width="1913" height="936" alt="Screenshot 2026-06-17 134940" src="https://github.com/user-attachments/assets/122ecb5e-8d71-461e-ab53-0271d88ec43e" />|<img width="1912" height="941" alt="Screenshot 2026-06-17 135017" src="https://github.com/user-attachments/assets/5df125bd-cf69-4707-8a56-8ff1bfba3cdb" />|<img width="1914" height="940" alt="Screenshot 2026-06-17 135038" src="https://github.com/user-attachments/assets/6ad4bd10-d370-4bfd-b2c8-5555d3719ef3" />|
|---------|---------|---------|

|<img width="1916" height="939" alt="Screenshot 2026-06-17 135104" src="https://github.com/user-attachments/assets/1d7c487a-eca4-47d4-a849-49079d525d7d" />|<img width="1911" height="937" alt="Screenshot 2026-06-17 135357" src="https://github.com/user-attachments/assets/485e6a5a-5e84-4d75-b447-5c98128aa6d3" />|<img width="1918" height="938" alt="Screenshot 2026-06-17 135713" src="https://github.com/user-attachments/assets/e67b411d-534d-4254-8057-772573397a54" />|
|---------|---------|---------|

### 🎓 xGATES Dashboard (Learner)

|<img width="1872" height="949" alt="Screenshot 2026-06-17 140454" src="https://github.com/user-attachments/assets/d75147eb-f3d6-4d36-a934-62d41c01868e" />|<img width="1872" height="949" alt="Screenshot 2026-06-17 140511" src="https://github.com/user-attachments/assets/1575b3e2-976f-4d9e-b6dc-66698d68e5f3" />|<img width="1865" height="948" alt="Screenshot 2026-06-17 140619" src="https://github.com/user-attachments/assets/2064dce1-1130-45bd-ab22-711a481b6fb3" />|
|---------|---------|---------|

|<img width="1868" height="951" alt="Screenshot 2026-06-17 141042" src="https://github.com/user-attachments/assets/505bf30c-3cbc-4d2f-b406-24f220850a39" />|<img width="1870" height="952" alt="Screenshot 2026-06-17 141053" src="https://github.com/user-attachments/assets/d30625d6-6517-49ce-b3de-2f6bdfd295ee" />|<img width="1871" height="945" alt="Screenshot 2026-06-17 141140" src="https://github.com/user-attachments/assets/04abd3ab-8b06-4b4b-83b8-c6d7de6df81d" />|
|---------|---------|---------|

|<img width="1874" height="952" alt="Screenshot 2026-06-17 141158" src="https://github.com/user-attachments/assets/c0889ddc-668b-4b77-83aa-a772b9e34e5f" />|<img width="1872" height="948" alt="Screenshot 2026-06-17 141230" src="https://github.com/user-attachments/assets/486b1368-f18c-4488-b223-d7b240910aa8" />|<img width="1866" height="953" alt="Screenshot 2026-06-17 141446" src="https://github.com/user-attachments/assets/b2dda383-25c9-4574-9363-8b28ca80d071" />|
|---------|---------|---------|

|<img width="1876" height="952" alt="Screenshot 2026-06-17 141510" src="https://github.com/user-attachments/assets/281d140d-768a-4677-a5c5-3b7ab47e57fb" />|<img width="1866" height="944" alt="Screenshot 2026-06-17 141623" src="https://github.com/user-attachments/assets/f41317dc-3468-492e-9758-9ae995e62c0a" />|<img width="1868" height="945" alt="Screenshot 2026-06-17 141642" src="https://github.com/user-attachments/assets/6c5823e5-aff7-435e-8cbc-9dd927a7529b" />|
|---------|---------|---------|

|<img width="1866" height="950" alt="Screenshot 2026-06-17 142123" src="https://github.com/user-attachments/assets/a0ed254f-761e-4bc2-88c9-b3fd8d302a87" />|<img width="1870" height="951" alt="Screenshot 2026-06-17 142220" src="https://github.com/user-attachments/assets/f04c174a-d603-4317-9ddb-b756829c2f4c" />|<img width="1909" height="940" alt="Screenshot 2026-06-17 142235" src="https://github.com/user-attachments/assets/784cf84c-1bbd-4dda-a976-5d6139eb2d17" />|
|---------|---------|---------|

|<img width="1869" height="947" alt="Screenshot 2026-06-17 142254" src="https://github.com/user-attachments/assets/8d5ddbcf-8809-43b6-9c6f-21ea419306ec" />|<img width="1908" height="939" alt="Screenshot 2026-06-17 142410" src="https://github.com/user-attachments/assets/0e6a2093-79ab-452e-8ba6-5e3d6e307d58" />|<img width="1910" height="939" alt="Screenshot 2026-06-17 142447" src="https://github.com/user-attachments/assets/6d72e337-c7c4-43ce-abb2-1090830bb6df" />|
|---------|---------|---------|

|<img width="1867" height="947" alt="Screenshot 2026-06-17 142541" src="https://github.com/user-attachments/assets/d4b85663-b794-4988-bdf6-805c97fdc783" />|<img width="1871" height="943" alt="Screenshot 2026-06-17 142553" src="https://github.com/user-attachments/assets/250a54d5-6355-48c7-b50e-f850e3992146" />|<img width="1864" height="949" alt="Screenshot 2026-06-17 142713" src="https://github.com/user-attachments/assets/b52d4a4b-eece-464d-a3e7-c0584e7b5f98" />|
|---------|---------|---------|

|<img width="1866" height="950" alt="Screenshot 2026-06-17 142734" src="https://github.com/user-attachments/assets/73845d8f-369c-4eef-bbac-2714ec69fe21" />|<img width="1865" height="951" alt="Screenshot 2026-06-17 142823" src="https://github.com/user-attachments/assets/0e09f324-03fe-48d6-9f1a-9fcbff7ddcc1" />|<img width="1866" height="948" alt="Screenshot 2026-06-17 142908" src="https://github.com/user-attachments/assets/739bd886-725a-45f6-9388-f427f926d0d9" />|
|---------|---------|---------|

|<img width="1867" height="950" alt="Screenshot 2026-06-17 142932" src="https://github.com/user-attachments/assets/09536c39-5db0-4fcc-87c8-96dc74c5939c" />|<img width="1865" height="952" alt="Screenshot 2026-06-17 143101" src="https://github.com/user-attachments/assets/58cd7783-96e6-4428-80fe-0695e0beeb2d" />|<img width="1865" height="949" alt="Screenshot 2026-06-17 143303" src="https://github.com/user-attachments/assets/ca2c6ac2-ae7e-4658-88d6-5588c9f95a64" />|
|---------|---------|---------|

|<img width="1867" height="951" alt="Screenshot 2026-06-17 143504" src="https://github.com/user-attachments/assets/ac6aaae6-981d-4476-9197-421d560d04e6" />|<img width="1867" height="948" alt="Screenshot 2026-06-17 143710" src="https://github.com/user-attachments/assets/8607a71f-3081-4226-bce2-dba14e7875f3" />|<img width="1864" height="954" alt="Screenshot 2026-06-17 143632" src="https://github.com/user-attachments/assets/c2ff659d-9d8b-4791-8931-1b29da6d0761" />|
|---------|---------|---------|


---






⚡️ [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)


