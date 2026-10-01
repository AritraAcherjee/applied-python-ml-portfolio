# Smart Travel Assistant

A LangChain mini-project that demonstrates tool use, conversation memory, and parallel LCEL chains.

## Assignment requirements covered

| Requirement | Where it is implemented |
|---|---|
| Accept travel-related user queries | `run_chat()` in `main.py` |
| Simulated Weather Tool | `get_weather()` + `WeatherTool` |
| Simulated Currency Tool | `convert_currency()` + `CurrencyTool` |
| Track conversation history | `ConversationBufferMemory` |
| Answer follow-up questions using memory | Agent prompt includes `{chat_history}` |
| LangChain agent | `initialize_agent(...)` |
| `ZERO_SHOT_REACT_DESCRIPTION` | `AgentType.ZERO_SHOT_REACT_DESCRIPTION` |
| Tourist-attraction chain | `attractions_chain` |
| Budget-advice chain | `budget_chain` |
| `RunnableSequence` | Used for both recommendation chains |
| `RunnableParallel` | `parallel_travel_chain` |
| OpenAI API key from `.env` | `python-dotenv` + `OPENAI_API_KEY` |
| Demonstration queries | `/demo` or manual commands |

## Project files

```text
smart-travel-assistant/
├── main.py
├── simple_version.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

`main.py` is the polished submission version.

`simple_version.py` is a shorter version that is easier to explain during a demo or presentation. It still shows all of the major required LangChain concepts.

## Setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create the `.env` file

Copy `.env.example` and rename the copy to `.env`.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Open `.env` and replace:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

with your real OpenAI API key.

**Do not upload `.env` to GitHub.** The included `.gitignore` prevents it from being committed.

### 4. Run the project

```bash
python main.py
```

## Commands

Inside the program:

```text
/plan Paris
/history
/demo
/quit
```

You can also type normal questions, for example:

```text
What is the weather in Paris?
Convert 100 USD to EUR.
What did I ask you first?
```

For the parallel-chain requirement:

```text
/plan Paris
```

The program generates attractions and budget advice using `RunnableParallel`.

## Suggested demonstration

Use this sequence when showing the project:

1. Run `python main.py`.
2. Ask: `What is the weather in Paris?`
3. Ask: `Convert 100 USD to EUR.`
4. Ask: `What did I ask you first?`
5. Enter: `/plan Paris`
6. Show that both recommendations and budget advice are produced.
7. Enter `/history` to show stored conversation history.

## Simulated data notice

The Weather Tool and Currency Tool intentionally use predefined/simulated values because the assignment asks for simulated tools. They do **not** call live weather or exchange-rate APIs.

## GitHub submission

After testing the project:

```bash
git init
git add .
git commit -m "Complete Smart Travel Assistant"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Submit the GitHub repository link requested by the assignment.

## Notes about LangChain versions

This project intentionally uses the LangChain `0.3.x` API because the assignment specifically asks for legacy course concepts such as:

- `ConversationBufferMemory`
- `initialize_agent`
- `AgentType.ZERO_SHOT_REACT_DESCRIPTION`

Newer major LangChain releases may change or remove these interfaces, so `requirements.txt` keeps LangChain below `0.4`.
