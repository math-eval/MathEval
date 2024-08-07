from conversation import get_conv_template

def build_prompt(input_conversation, template_name, args):
    """
    Build a prompt based on the input conversation and template.

    Parameters:
    - input_conversation (list): A list of dictionaries representing the conversation.
    - template_name (str): The name of the template to use.
    - args (dict): Additional arguments.

    Returns:
    - str: The generated prompt.
    """
    conv = get_conv_template(template_name)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    
    # Apply prompt templates
    if roles[input_conversation[0]["from"]] != conv.roles[0]:
        input_conversation = input_conversation[1:]
    
    conv.messages = []
    for j, sentence in enumerate(input_conversation):
        role = roles[sentence['from']]
        assert role == conv.roles[j % 2]
        conv.append_message(role, sentence['value'])
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

# Example usage
if __name__ == "__main__":
    # Example input
    input_conversation = [
        {"from": "human", "value": "Hello!"},
        {"from": "gpt", "value": "Hi there! How can I help you today?"},
        {"from": "human", "value": "I need some information about AI."}
    ]
    template_name = "raw"
    args = {}

    # Call the function
    prompt = build_prompt(input_conversation, template_name, args)
    print(prompt)