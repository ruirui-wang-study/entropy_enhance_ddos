import numpy as np

def renyi_joint_entropy(alpha, joint_probs):
    # Ensure that alpha is not close to 1 to avoid division by zero
    if np.isclose(alpha, 1):
        raise ValueError("Alpha should not be close to 1.")
    
    # Calculate the Renyi joint entropy
    result = (1 / (1 - alpha)) * np.log(np.sum(joint_probs ** alpha))
    
    return result

# Example usage
# Replace joint_probs with your actual joint probability distribution
joint_probs = np.array([[0.2, 0.1], [0.1, 0.3]])
alpha_value = 0.5  # Replace with your desired alpha value

result = renyi_joint_entropy(alpha_value, joint_probs)
print("Renyi Joint Entropy:", result)
