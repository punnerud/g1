import streamlit as st
import requests
import json
import time
import re

def stream_api_call(messages, max_tokens, is_final_answer=False):
    prompt = json.dumps(messages)
    data = {
        "model": "llama3.1:8b",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True
    }
    try:
        response = requests.post('http://localhost:11434/api/generate', 
                                 headers={'Content-Type': 'application/json'}, 
                                 data=json.dumps(data),
                                 stream=True)
        response.raise_for_status()
        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                if 'response' in chunk:
                    full_response += chunk['response'].replace("'",'')
                    yield chunk['response'].replace("'",'')
        if full_response:
            return json.loads(full_response.replace("'",''))
        else:
            raise ValueError("Empty response from API")
    except Exception as e:
        error_message = f"Failed to generate {'final answer' if is_final_answer else 'step'}. Error: {str(e)}"
        return {"title": "Error", "content": error_message, "next_action": "final_answer"}

def extract_json(text):
    # Remove triple backticks and any language specifiers
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.strip()  # Remove leading/trailing whitespace
    
    #print("extract json:")
    #print(text)
    #print("\n\n\n")
    
    # Find all JSON-like substrings
    json_objects = re.findall(r'\{[^{}]*\}', text)
    
    if json_objects:
        # If we found JSON-like substrings, return the last one
        try:
            return json.loads(json_objects[-1])
        except json.JSONDecodeError:
            pass
    
    # If no valid JSON objects were found, return a default object
    return {
        "title": "Parsing Error",
        "content": text,
        "next_action": "continue"
    }

def generate_response(prompt):
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES."""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    # Create a single placeholder for streaming output
    streaming_output = st.empty()
    
    while True:
        start_time = time.time()
        step_data = ""
        for chunk in stream_api_call(messages, 300):
            step_data += chunk
            streaming_output.write(f"Step {step_count} (streaming):\n\n{step_data}")
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        step_json = extract_json(step_data)
        title = step_json.get('title', 'Untitled')
        content = step_json.get('content', 'No content')
        next_action = step_json.get('next_action', 'continue')
        
        steps.append((f"Step {step_count}: {title}", content, thinking_time))
        messages.append({"role": "assistant", "content": json.dumps(step_json)})
        
        if next_action == 'final_answer':
            break
        
        step_count += 1
        yield steps, None

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})
    
    start_time = time.time()
    final_data = ""
    for chunk in stream_api_call(messages, 200, is_final_answer=True):
        final_data += chunk
        streaming_output.write(f"Final answer (streaming):\n\n{final_data}")
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    final_json = extract_json(final_data)
    steps.append(("Final Answer", final_json.get('content', final_data), thinking_time))

    yield steps, total_thinking_time

def main():
    st.set_page_config(page_title="Local Llama Prototype", page_icon="🧠", layout="wide")
    
    st.title("Local Llama: Using Llama model locally to create reasoning chains")
    
    st.markdown("""
    This is an early prototype of using prompting to create reasoning chains to improve output accuracy. It is not perfect and accuracy has yet to be formally evaluated. It is powered by a locally hosted Llama model.
    """)
    
    user_query = st.text_input("Enter your query:", placeholder="e.g., How many 'R's are in the word strawberry?")
    
    if user_query:
        st.write("Generating response...")
        
        response_container = st.container()
        time_container = st.empty()
        
        previous_step_count = 0
        for steps, total_thinking_time in generate_response(user_query):
            with response_container:
                for i in range(previous_step_count, len(steps)):
                    title, content, thinking_time = steps[i]
                    if title.startswith("Final Answer"):
                        st.markdown(f"### {title}")
                        st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                    else:
                        with st.expander(title, expanded=True):
                            st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
            
            previous_step_count = len(steps)
            
            if total_thinking_time is not None:
                time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")

if __name__ == "__main__":
    main()