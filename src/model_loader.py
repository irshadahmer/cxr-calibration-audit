import torch
import torchxrayvision as xrv

def load_cxr_model(weights="densenet121-res224-chex"):
    """Loads clinical DenseNet-121 model trained on CheXpert."""
    model = xrv.models.DenseNet(weights=weights)
    model.eval()
    return model

def predict_batch(model, images_tensor):
    """Passes standard CXR images and returns logits + latent features."""
    with torch.no_grad():
        features = model.features(images_tensor)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        latent_features = torch.flatten(out, 1)
        logits = model.classifier(latent_features)
    return logits.cpu().numpy(), latent_features.cpu().numpy()
