## Setup

1. Install [ollama](https://github.com/ollama/ollama).
2. Test using `ollama run gemma3`. This step is going to take some time in the first instance since its going to pull a big model.

### ollama

Some helpful commands. 

```
ollama run "model"

ollama list 

ollama rm "model"
```


## TODO

1. Test out langraph UI. Document.
  - Can we see traces? 
2. Let's create a documentation learning agent. 
  - We ask it questions about langgraph and it has 2 tools, memory and search. 
  - If what we are asking is not in memory, it will search and put it in memory. 
    - Probably have section, subsection etc?
  - If already in memory, it will use that to base its answer. 
  - We should also take the time to update the prompt.
