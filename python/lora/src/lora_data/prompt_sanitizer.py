import os
import re


def is_technical_command(line: str) -> bool:
    line = line.strip().lower()
    chat_patterns = [
        "does all that research", "what? just the case", "add a guidance note",
        "and then why did you stop", "do you think this is going to work",
        "go search for the docs", "is the new plan better", "isn't your experiment plan",
        "you completely got rid", "no, you should make", "first update the plan",
        "now tell me again", "why doesn't it appear", "you just said to me",
        "is the training still", "is there some way to verify", "why why do you",
        "give me your plan", "that is when the plan stops", "why is the moonshine document",
        "let's actually reread", "does it end abruptly", "read again the guidance",
        "i don't want any tests", "now updatge @agents", "no i only want",
        "i asked a question", "is any thing even running", "when i go into visual",
        "do you have access", "now will you again", "why did you pick",
        "now look at the logs", "is that intentional", "after training is complete",
        "review @_posts", "can you add a training", "why do i not see",
        "considering how much", "i don't know what you are", "no edits at all",
        "create docs/personalized_training", "move @docs", "tell me again what",
        "you're just going around", "that is not what i want", "how would we interact",
        "so my description of", "what does stopping only", "you will run out of",
        "is there another theme", "why is vim fugitive", "would it help if",
        "is mermaid the best", "surely jekyll or", "why do you keep doing",
        "can you first update", "add your suggestions", "read @plan.md",
        "put a to do docstring", "review @refactor.md", "diagnosis",
        "can we do that as", "read docs/", "nope don't want this",
        "redo both experiment", "again, why", "include in @docs",
        "is @plan.md a better", "what is fcp", "write a voice.md",
        "use gh cli", "review @docs", "how would you configure",
        "poll yourself", "you did not keep your", "why are you so lazy",
        "and will you stop", "do not use streaming", "look @plan",
        "can you use the just", "rewrite plan.md", "wait, did you make changes",
        "look @docs", "why do you have to use", "yes, and do not",
        "what is the build manifest", "what is this going around",
        "is the revised @plan", "read the document again", "where are you seeing",
        "@docs/experiment_plan", "make the docs string", "read exp_",
        "i will remove it", "look at all the recent", "does the current blog",
        "write a detailed", "i have removed your", "no, do not add a fix",
        "write a verification", "did not ask you to", "does @plan.md account",
        "you can use timeout", "gf in vsplit", "in nvim, how can i",
        "you're not doing a very good", "why do you insist", "what do you think",
        "no files matching", "stop reading continuously", "did you stop monitoring",
        "that is not what i meant", "can you create a just file", "review again",
        "why do the number of samples", "now based on everything",
        "it seems to be plateauing", "read @tuning", "if a process is working",
        "look at all of our", "stop telling me things", "only update the docs",
        "look at the other posts", "and so the assumption", "write your review",
        "training should have been", "are there any tests", "we ran all these",
        "update the experiment plan", "did you just stop", "don't do head",
        "make sure arguments", "continue. you are trying", "let's stay on dora",
        "i don't think the job", "can i also get", "will this achieve what",
        "can you stick to using", "why do you think", "actually forget that",
        "is there some way to figure", "that moonshine paper", "is that consistent with",
        "what is the correct way", "no, you will not stop", "i don't see most",
        "if we continue down", "system: please continue", "didn't we agree to",
        "i didn't see you download", "merge all make files", "you have the capability",
        "based on how the training", "based on everything you", "polling should be via",
        "you're not polling", "based on your moonshine", "now think hard",
        "no, i even want the ux", "what if i'm already", "okay, start and do not stop",
        "at least put it as documentation", "are the json files generated",
        "but we specifically want", "are any of your changes", "look @outputs",
        "content from", "fix the typose", "okay", "no edits", "i fixed somethings",
        "i asked a question", "is any thing", "put this in", "when i go", "do you have access",
        "now will you again", "why did you pick", "is that intentional",
        "first tell me", "what is the correct way", "i asked you to debug"
    ]
    for pattern in chat_patterns:
        if pattern in line:
            return False
    return True

def sanitize_prompts(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        print(f"Warning: {input_path} not found.")
        return
        
    with open(input_path) as f:
        lines = f.readlines()
        
    clean_commands = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if is_technical_command(line):
            cleaned = re.sub(r'^\d+\.\s*', '', line)
            clean_commands.append(cleaned)
            
    with open(output_path, 'w') as f:
        for cmd in set(clean_commands):
            f.write(cmd + "\n")
            
    print(f"Sanitized {len(set(clean_commands))} commands to {output_path}")

if __name__ == "__main__":
    sanitize_prompts("docs/personalized/prompts.txt", "data/coding_prompts.txt")
