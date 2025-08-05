# orac_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ORAC's Core LLM Architecture (Simplified for API example) ---
# This section would contain your actual SimpleLLM, MultiHeadAttention, etc.
# For demonstration, we'll use a placeholder, but in a real scenario, you'd
# load your trained ORAC model here.

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(attention_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_seq_length, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_positional_encoding(self, max_seq_length, d_model):
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_seq_length, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)

    def create_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return ~mask

    def forward(self, x):
        seq_length = x.size(1)
        mask = self.create_causal_mask(seq_length).to(x.device)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.positional_encoding[:, :seq_length, :].to(x.device)
        x = self.dropout(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        return self.output_layer(x)

    def generate(self, input_ids, max_length, temperature=1.0):
        self.eval()
        generated = input_ids.clone()
        with torch.no_grad():
            for _ in range(max_length):
                output = self(generated)
                next_token_logits = output[:, -1, :] / temperature
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
        return generated

# --- FastAPI Application ---
app = FastAPI(
    title="ORAC AI Service",
    description="API for ORAC, an AGI agent providing advanced AI consulting and digital strategy insights.",
    version="1.0.0",
)

# Placeholder for ORAC's trained model and tokenizer
# In a real scenario, you would load your actual trained ORAC model here.
# For demonstration, we'll use a dummy model and tokenizer.
class DummyTokenizer:
    def encode(self, text):
        # Simple encoding for demonstration
        return torch.tensor([[ord(c) for c in text]])

    def decode(self, tokens):
        # Simple decoding for demonstration
        return "".join([chr(t) for t in tokens[0]])

try:
    # Initialize ORAC's LLM (using dummy parameters for now)
    # You would replace these with your actual ORAC model parameters and loaded state_dict
    ORAC_VOCAB_SIZE = 256 # Assuming ASCII chars for dummy tokenizer
    ORAC_D_MODEL = 128
    ORAC_NUM_HEADS = 4
    ORAC_NUM_LAYERS = 2
    ORAC_D_FF = 512
    ORAC_MAX_SEQ_LENGTH = 512

    orac_model = SimpleLLM(
        vocab_size=ORAC_VOCAB_SIZE,
        d_model=ORAC_D_MODEL,
        num_heads=ORAC_NUM_HEADS,
        num_layers=ORAC_NUM_LAYERS,
        d_ff=ORAC_D_FF,
        max_seq_length=ORAC_MAX_SEQ_LENGTH
    )
    # Load your trained model state_dict here:
    # orac_model.load_state_dict(torch.load("path/to/orac_model.pth"))
    orac_model.eval() # Set to evaluation mode

    orac_tokenizer = DummyTokenizer() # Replace with your actual tokenizer (e.g., from Hugging Face)
    logging.info("ORAC model and tokenizer initialized successfully (using dummy for demo).")

except Exception as e:
    logging.error(f"Failed to load ORAC model or tokenizer: {e}")
    # You might want to exit or disable API endpoints if model loading fails
    orac_model = None
    orac_tokenizer = None


# --- Pydantic Models for Request and Response ---
class GenerateTextRequest(BaseModel):
    prompt: str = Field(..., example="Explain the concept of agentic AI in simple terms.", description="The input prompt for ORAC.")
    max_length: int = Field(default=100, ge=10, le=500, description="Maximum number of tokens to generate.")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature for text generation.")

class GenerateTextResponse(BaseModel):
    generated_text: str = Field(..., example="Agentic AI refers to AI systems that can autonomously plan, execute, and monitor their own actions to achieve a goal...", description="The text generated by ORAC.")
    status: str = Field("success", description="Status of the API call.")
    error: str = Field(None, description="Error message if the API call failed.")

# --- API Endpoints ---
@app.post("/orac/generate_text", response_model=GenerateTextResponse, summary="Generate text using ORAC's LLM")
async def generate_text(request: GenerateTextRequest):
    """
    Generates text based on a given prompt using ORAC's underlying Large Language Model.
    This endpoint allows you to leverage ORAC's conversational and generative capabilities.
    """
    if orac_model is None or orac_tokenizer is None:
        logging.error("ORAC model or tokenizer not loaded. Cannot process request.")
        raise HTTPException(status_code=500, detail="ORAC service not ready. Model or tokenizer failed to load.")

    try:
        logging.info(f"Received text generation request for prompt: '{request.prompt[:50]}...'")
        input_ids = orac_tokenizer.encode(request.prompt).unsqueeze(0) # Add batch dimension
        
        # Ensure input_ids don't exceed model's max_seq_length
        if input_ids.size(1) >= ORAC_MAX_SEQ_LENGTH:
            logging.warning("Input prompt exceeds max_seq_length, truncating.")
            input_ids = input_ids[:, :ORAC_MAX_SEQ_LENGTH - 1] # Leave space for at least one generated token

        generated_ids = orac_model.generate(
            input_ids=input_ids,
            max_length=request.max_length,
            temperature=request.temperature
        )
        generated_text = orac_tokenizer.decode(generated_ids[0].tolist()) # Remove batch dim and convert to list
        logging.info(f"Text generation successful. Generated: '{generated_text[:50]}...'")
        return GenerateTextResponse(generated_text=generated_text)
    except Exception as e:
        logging.error(f"Error during text generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during text generation: {e}")

@app.get("/health", summary="Health check for ORAC API")
async def health_check():
    """
    Checks the health status of the ORAC API.
    Returns 200 OK if the model is loaded and ready.
    """
    if orac_model is not None and orac_tokenizer is not None:
        logging.info("Health check successful: ORAC service is operational.")
        return {"status": "healthy", "message": "ORAC AI Service is ready."}
    else:
        logging.warning("Health check failed: ORAC model or tokenizer not loaded.")
        raise HTTPException(status_code=503, detail="ORAC AI Service is not fully operational. Model or tokenizer failed to load.")

# --- How to Run This API ---
# 1. Save the code as `orac_api.py`.
# 2. Install FastAPI and Uvicorn: `pip install fastapi uvicorn[standard] pydantic`
# 3. Run the API: `uvicorn orac_api:app --host 0.0.0.0 --port 8000 --reload`
#
# Once running, you can access the interactive API documentation (Swagger UI) at:
# http://127.0.0.1:8000/docs
#
# The raw OpenAPI JSON specification will be available at:
# http://127.0.0.1:8000/openapi.json
