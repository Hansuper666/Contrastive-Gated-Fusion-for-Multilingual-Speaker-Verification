import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """
    Gated fusion from baseline - transforms audio and image embeddings
    Returns fused output AND transformed audio/image embeddings
    """
    def __init__(self, embed_dim, mid_att_dim=128):
        super(GatedFusion, self).__init__()
        
        self.linear_audio = nn.Linear(embed_dim, embed_dim)
        self.linear_image = nn.Linear(embed_dim, embed_dim)
        
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 2, mid_att_dim),
            nn.BatchNorm1d(mid_att_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(mid_att_dim, embed_dim)
        )
    
    def forward(self, audio_input, image_input):
        """
        Args:
            audio_input: (batch_size, embed_dim)
            image_input: (batch_size, embed_dim)
        
        Returns:
            fused: (batch_size, embed_dim) - gated fusion output
            audio_trans: (batch_size, embed_dim) - transformed audio
            image_trans: (batch_size, embed_dim) - transformed image
        """
        concat = torch.cat([audio_input, image_input], dim=1)
        attention_out = torch.sigmoid(self.attention(concat))
        
        audio_trans = torch.tanh(self.linear_audio(audio_input))
        image_trans = torch.tanh(self.linear_image(image_input))
        
        # Gated combination
        fused = audio_trans * attention_out + image_trans * (1.0 - attention_out)
        
        return fused, audio_trans, image_trans


class CrossModalVerificationModel(nn.Module):
    """
    Cross-modal speaker verification model
    Architecture follows the diagram with separate branches for audio and image
    With proper normalization layers added
    """
    def __init__(
        self,
        audio_input_dim=1024,
        image_input_dim=1536,
        hidden_dim=512,
        embed_dim=256,
        num_classes=50,
        dropout=0.5,
        activation='gelu'
    ):
        super(CrossModalVerificationModel, self).__init__()
        
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        
        # Input normalization for pre-encoded features
        self.audio_input_norm = nn.LayerNorm(audio_input_dim)
        self.image_input_norm = nn.LayerNorm(image_input_dim)
        
        # Audio branch: LayerNorm -> FC -> BatchNorm -> Activation -> Dropout -> FC -> BatchNorm -> Activation
        self.audio_branch = nn.Sequential(
            nn.Linear(audio_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU()
        )
        
        # Image branch: LayerNorm -> FC -> BatchNorm -> Activation -> Dropout -> FC -> BatchNorm -> Activation
        self.image_branch = nn.Sequential(
            nn.Linear(image_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU()
        )
        
        # Gated fusion layer (like baseline)
        self.gated_fusion = GatedFusion(embed_dim=embed_dim, mid_att_dim=128)
        
        # Classification layer
        self.logits_layer = nn.Linear(embed_dim, num_classes)
    
    def forward(self, audio_features, image_features):
        """
        Args:
            audio_features: (batch_size, audio_input_dim)
            image_features: (batch_size, image_input_dim)
        
        Returns:
            logits: (batch_size, num_classes) for classification
            audio_trans: (batch_size, embed_dim) transformed audio for verification
            image_trans: (batch_size, embed_dim) transformed image for verification
            fused: (batch_size, embed_dim) for contrastive loss
        """
        # Normalize input features first
        audio_features = self.audio_input_norm(audio_features)
        image_features = self.image_input_norm(image_features)
        
        # Process through separate branches
        audio_embed_raw = self.audio_branch(audio_features)
        image_embed_raw = self.image_branch(image_features)
        
        # Normalize branch outputs (like baseline EmbedBranch line 74)
        audio_embed = F.normalize(audio_embed_raw, p=2, dim=1)
        image_embed = F.normalize(image_embed_raw, p=2, dim=1)
        
        # Gated fusion (like baseline)
        # Returns fused features AND transformed audio/image
        fused, audio_trans, image_trans = self.gated_fusion(audio_embed, image_embed)
        
        # Classification
        logits = self.logits_layer(fused)
        
        return logits, audio_trans, image_trans, fused
    
    def get_embeddings(self, audio_features, image_features):
        """Extract transformed embeddings from fusion for verification"""
        # Normalize input features first
        audio_features = self.audio_input_norm(audio_features)
        image_features = self.image_input_norm(image_features)
        
        audio_embed_raw = self.audio_branch(audio_features)
        image_embed_raw = self.image_branch(image_features)
        
        # Normalize
        audio_embed = F.normalize(audio_embed_raw, p=2, dim=1)
        image_embed = F.normalize(image_embed_raw, p=2, dim=1)
        
        # Get transformed embeddings from fusion
        _, audio_trans, image_trans = self.gated_fusion(audio_embed, image_embed)
        
        return audio_trans, image_trans

