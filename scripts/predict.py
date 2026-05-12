import torch
import numpy as np
from experiment import STLModel, MTLModel, generate_student_data
from sklearn.preprocessing import StandardScaler

# 1. Load Data & Scaler (for consistent preprocessing)
X, y_reg, y_cls = generate_student_data(n_samples=100)
scaler = StandardScaler()
scaler.fit(X) # In a real project, we'd save the scaler, but here we re-fit for demo

# 2. Load Models
stl = STLModel()
stl.load_state_dict(torch.load('models/stl_model.pth'))
stl.eval()

mtl = MTLModel()
mtl.load_state_dict(torch.load('models/mtl_model.pth'))
mtl.eval()

# 3. Create a Sample Student
# High study time (feat 0), High attendance (feat 1)
sample_student = np.array([[0.9, 0.9, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
sample_student_t = torch.FloatTensor(scaler.transform(sample_student))

# 4. Make Predictions
print("\n--- Model Inference Demo ---")
with torch.no_grad():
    stl_score = stl(sample_student_t).item()
    mtl_score, mtl_pass_prob = mtl(sample_student_t)
    
    print(f"Sample Student Features: Study=90%, Attendance=90%")
    print(f"STL Predicted Score: {stl_score:.2f}/100")
    print(f"MTL Predicted Score: {mtl_score.item():.2f}/100")
    print(f"MTL Pass Probability: {mtl_pass_prob.item()*100:.1f}%")
    print(f"MTL Final Status: {'PASS' if mtl_pass_prob.item() > 0.5 else 'FAIL'}")
