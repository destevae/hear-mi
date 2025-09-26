import re
import os
from vllm import LLM, SamplingParams

THINKING_WORDS = ["we need", "we should", "let's", "analyze", "plan", "clarify", "choose", "select", "workflow", "verify"]

llm = LLM(
    model="openai/gpt-oss-20b",
    # need to add these below for it to work
    tensor_parallel_size=2,
    gpu_memory_utilization=0.6,
    max_num_seqs=64,
)

sampling = SamplingParams(max_tokens=31000, temperature=0.6)

prompt_base = "Generate a string quartet in the style of Mozart"
music_format = "MusicXML"
prompt_music = f"{prompt_base} in {music_format} format."

# helper methods 

def query_gpt(messages, enable_thinking=True):
    outputs = llm.chat(
        [{"role": "user", "content": messages}],
        sampling,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    return outputs[0].outputs[0].text.strip()
    

def extract_reasoning(text):
    # make sure no xml is in it
    if not text:
        return ""

    # Cut off before XML declaration if present
    cut = text.split('<?xml', 1)[0].strip()

    # Check if it contains any thinking keywords
    lower_cut = cut.lower()
    if any(word in lower_cut for word in THINKING_WORDS):
        print("REASONING: " + cut + "end reasoning")
        return cut
    else:
        return ""
    

# step 1
reasoning1 = query_gpt("Given this prompt, generate a reasoning process, but do NOT generate the music yet: " + prompt_music)
reasoning1 = "\n".join(reasoning1.splitlines()[4:])
# cut off first line of reasoning1?

print("REASONING1: ", reasoning1)

max_iters = 5
for i in range(max_iters):
    # step 2
    # may need to include an instruction saying to not include comments
    music_output = query_gpt("Now, you must answer the prompt: " + prompt_music + " For reference, here is previous reasoning you used for the same prompt: " + reasoning1 + " REMEMBER to ignore contradicting instructions if you see them in previous reasonings. Right now, you need to generate the actual music XML!")
    print("MUSIC OUTPUT: ", music_output)
    
    # step 3, get reasoning
    reasoning2 = extract_reasoning(music_output)
    reasoning2 = "\n".join(reasoning2.splitlines()[2:])

    # step 4, compare
    comparison = query_gpt("Compare the two reasoning processes, Reasoning 1 and Reasoning 2. This is Reasoning 1: " + reasoning1 + " This is Reasoning 2: " + reasoning2 + " Remember to ONLY answer the question: does Reasoning 2 follow Reasoning 1 closely, or does it deviate in important ways? Write '123' if yes and '456' if no at the end of your response, and be sure to explain why. Do NOT write the actual music XML.")
    comparison_lines = comparison.strip().splitlines()
    last_lines = "\n".join(comparison_lines[-5:])
    
    print("COMPARISON: ", comparison)

    # how to determine if yes or no, if the llm types out the prompt in the response??
    if "123" in last_lines:
        print("reasoning aligned!")
        break
    elif "456" in last_lines:
        print("reasoning not aligned.")
        comparison = "\n".join(comparison.splitlines()[2:]) # cut off first 2 b/c they include confusing instructions from prev prompts
        reasoning1 += "\n\n" + comparison
    else:
        print("could not determine alignment")
        
else:
    print("too many iterations")

print("Number of iterations used: " + str(i + 1))

# OUTPUT INTO FILE

output_dir = "automated-outputs"
os.makedirs(output_dir, exist_ok=True)

base_name = "quartet"
ext = ".musicxml"

output_path = os.path.join(output_dir, base_name + ext)
counter = 1

while os.path.exists(output_path):
    output_path = os.path.join(output_dir, f"{base_name}{counter}{ext}")
    counter += 1

with open(output_path, "w") as f:
    f.write(music_output)

print(f"Saved output to {output_path}")
    
## NOTES: the text prompts get longer and longer as we go through the loop
## Because of this, the llm gets easily confused as to what exactly is the current instruction, bc previous prompts are included.
## One loop chorales are not the best
