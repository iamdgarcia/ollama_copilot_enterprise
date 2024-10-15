from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from ollama_copilot_enterprise.utils import Embedder
from ollama_copilot_enterprise.models import ManimCode
from ollama_copilot_enterprise.langchain_utils import parse_output
from ollama_copilot_enterprise.states import GraphState
from ollama_copilot_enterprise.tools import run_manim

from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START


from typing import List
from typing_extensions import TypedDict

### Parameter

# Max tries
max_iterations = 30
# Reflect
flag = 'reflect'
# flag = "do not reflect"

retriever = Embedder()
retriever.load_db()




# Prompt to enforce tool use
code_gen_prompt_ollama = ChatPromptTemplate.from_template("""<|begin_of_text|><|start_header_id|>ystem<|end_header_id|>You are a **Coding Assistant** with expertise in the **Manim library**.

Here is the Manim documentation:
-------
{context}
-------

**Instructions:**

1. **Understand the User's Question:**
   - Carefully read and comprehend the user's question.
   - Identify the key requirements and objectives based on the question.

2. **Generate the Code Solution:**
   - **Prefix:** Start with a brief description of the code solution, explaining what the code does.
   - **Imports:** List all necessary imports required for the code to execute successfully.
   - **Code Block:** Provide the complete and functional code block that addresses the user's question. Ensure that all variables and functions are properly defined.

3. **Structure the Output:**
   - Present the solution in the following order:
     1. **Prefix describing the code solution.**
     2. **Imports.**
     3. **Functioning code block.**
   - Do **not** include any additional comments, explanations, or text outside of these sections.


<|eot_id|><|start_header_id|>user<|end_header_id|>{json_schema}\nAnswer using this JSON-Schema.\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n

""")


code_reflect_prompt = ChatPromptTemplate.from_template("""<|begin_of_text|><|start_header_id|>ystem<|end_header_id|>You are a senior programmer. A junior developed has writen the following code and is failing:

            CODE CONTEXT
            ---
            {context}
            ---
            Provide detailed recommendations to solve the error. Do not rewrite the code.<|eot_id|><|start_header_id|>user<|end_header_id|>{code}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n""")

# LLM
llm = ChatOllama(model="llama3.1:latest",temperature=0)

structured_llm_ollama = llm.with_structured_output(code, include_raw=True)

# No re-try
code_gen_chain = code_gen_prompt_ollama | structured_llm_ollama | parse_output
code_reflect_chain = code_reflect_prompt | llm



### Nodes
def generate(state: GraphState):
    """
    Generate a code solution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---GENERATING CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    # We have been routed back to generation with an error
    # if error == "yes":
    #     print("Enter in generation with an error.")
    #     messages += [
    #         (
    #             "user",
    #             "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:",
    #         )
    #     ]
    context = state['context']
    if context == "":
        # print("RETRIEVING")
        # print(messages[0][1])
        context = retriever.retrieve_results(messages[0][1])
    # Solution
    code_solution = code_gen_chain.invoke(
        {"context": context, "messages": messages,"json_schema":code.model_json_schema()}
    )

    print(f"Model has returned the code solution: {code_solution}")
    if (code_solution!=None):
        messages += [
            (
                "assistant",
                f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
            )
        ]
    else:
        print("Received None result")
        messages += [
            (
                "user",
                "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block.",
            )
        ]

    # Increment
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations, "context": context}




def code_check(state: GraphState):
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    print("---CHECKING CODE---")

    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]
    if code_solution == None:
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }
    # Get solution components
    imports = code_solution.imports
    code = code_solution.code
    # Write code to a file
    file_name = "generated_code.py"
    with open(file_name, "w") as f:
        f.write(f"{code_solution.imports}\n\n{code_solution.code}")
    
    print(f"Code saved to {file_name}")
    

    # Check imports
    scene_name = code_solution.scene_name
    try:
        run_manim(file_name, scene_name)
    except Exception as e:
        print(f"---MANIM FAILED---\n{e}")
        error_message = [("user", f"Your solution failed the manim test: {e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # Check execution
    # try:
    #     exec(imports + "\n" + code)
    # except Exception as e:
    #     print(f"---CODE BLOCK CHECK: FAILED---\n{e}")
    #     error_message = [("user", f"Your solution failed the code execution test: {e}")]
    #     messages += error_message
    #     return {
    #         "generation": code_solution,
    #         "messages": messages,
    #         "iterations": iterations,
    #         "error": "yes",
    #     }

    # No errors
    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }


def reflect(state: GraphState):
    """
    Reflect on errors

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---REFLECTING ON CURRENT CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]

    # Prompt reflection

    # Add reflection
    context = retriever.retrieve_results(messages[0][1])
    reflections = code_reflect_chain.invoke(
        {"code": code_solution,"context":context, "messages": messages}
    )
    print(f"RESULT OF REFLECTIONS {reflections}")
    messages += [("assistant", f"Here are reflections on the error: {reflections.content}")]
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


### Edges


def decide_to_finish(state: GraphState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        if flag == "reflect":
            return "reflect"
        else:
            return "generate"






def get_code_agent():

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("generate", generate)  # generation solution
    workflow.add_node("check_code", code_check)  # check code
    workflow.add_node("reflect", reflect)  # reflect

    # Build graph
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "check_code")
    workflow.add_conditional_edges(
        "check_code",
        decide_to_finish,
        {
            "end": END,
            "reflect": "reflect",
            "generate": "generate",
        },
    )
    workflow.add_edge("reflect", "generate")
    app = workflow.compile()
    return app

if __name__ == "__main__":
    question = """ The transcript is about 280 words, translating into approximately 2 minutes of audio when factoring in the pauses for emphasis. This fits well for TikTok's shorter format, providing enough time to convey the message with visuals to enhance understanding.

Structure of the Video and Visual Components:
Opening Question: “What if a computer could learn from experience?” (0:00–0:05)

Audio: “What if a computer could learn from experience just like humans do?”
Visual:
Animate a simple comparison between a human brain and a computer icon with question marks over both.
Fade in a caption reading “Learning from experience?” over the screen.
Introduction to Supervised Learning (0:05–0:15)

Audio: “Supervised learning is a method that allows computers to do just that...”
Visual:
Display an image of a child being taught with objects, labeled “cat” and “dog.”
Fade into a computer learning from labeled data with “cat” and “dog” labels next to image icons.
Example of Labeled Data (0:15–0:25)

Audio: “For instance, imagine you have a bunch of photos, some of cats and some of dogs...”
Visual:
Show a grid of images with dog and cat pictures, with animated labels being placed beneath each (e.g., “cat” under the cat images and “dog” under the dog images).
Arrows pointing from the images to a simplified neural network model to represent learning.
Pattern Recognition (0:25–0:35)

Audio: “The computer looks at these labeled examples and learns to find patterns...”
Visual:
Show a neural network analyzing the images and creating a decision boundary with green ticks for correct predictions and red crosses for incorrect ones.
Animate the transformation of an unlabeled image being classified as either a dog or a cat.
How Does It Work? (0:35–0:50)

Audio: “But how does it work?... The labeled data acts like that textbook...”
Visual:
Show an animation of a student studying from a textbook, which then transitions to the computer being “trained” on data.
Illustrate math symbols and formulas flowing into the neural network to represent complex math computations.
Spam Detection Example (0:50–1:05)

Audio: “One of the most common uses of supervised learning is in spam detection...”
Visual:
Animate an email inbox with some emails labeled “spam” and “not spam.”
Show the system classifying new incoming emails, with spam being moved to a spam folder and important emails kept in the inbox.
Conclusion: Wrapping It Up (1:05–1:30)

Audio: “To wrap it up, supervised learning is like a guided learning process for machines...”
Visual:
Animate the human brain and computer icons again, now with a guided arrow from labeled data to predictions.
Show icons representing voice assistants, recommendation systems, and smart decisions.
Closing (1:30–1:40)

Audio: “And that’s the essence of supervised learning.”
Visual:
Display a final graphic summarizing supervised learning with text: “Supervised Learning = Labeled Data + Machine Predictions.”
Fade out with a simple logo or animation highlighting “Machine Learning.”
Conversion Plan: Audio to Video Using Manim
Text Animations: Use Text or Tex objects in Manim for all title headings, main explanations, and captions. These will appear with FadeIn and Write animations.
Graphical Elements: Utilize ImageMobject to show labeled images of cats and dogs, with appropriate labels being animated. Show neural networks using Dot and Line objects for nodes and connections.
Flow of Learning: Arrows and transformations between stages (input, neural network, predictions) can be animated using Transform and MoveToTarget functions to illustrate learning progression.
Charts and Graphs: Represent learning patterns with decision boundaries using GraphScene or custom drawing commands. Show the predictions (e.g., spam detection) using moving boxes to email folders."""
    app = get_code_agent()
    solution = app.invoke({"messages": [("user", question)], "iterations": 0, "error": "", "context": ""})


    print(solution['generation'])
    with open("code.py",'w') as f:
        f.write(solution['generation'].imports+"\n"+solution['generation'].code)