# Well Actually API Documentation

A comprehensive REST API for evaluating software repositories in the Ethereum ecosystem using multiple LLM-based assessment tasks.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication required. Set `OPENROUTER_API_KEY` environment variable on server.

## Content Type
All requests and responses use `application/json`.

---

## Endpoints Overview

| Endpoint | Method | Purpose |
|----------|---------|---------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/compare` | POST | Repository comparison (L1/L3) |
| `/assess` | POST | Originality assessment (L2) |
| `/criteria` | POST | Criteria-based evaluation |
| `/special-case/status` | GET | Special case comparison status |
| `/special-case/toggle` | POST | Enable/disable special cases |
| `/cache/stats` | GET | Cache statistics |
| `/debug/comparison-methods` | GET | Available comparison methods |

---

## Core Endpoints

### 1. Repository Comparison

**Endpoint:** `POST /compare`

**Purpose:** Compare two repositories for their relative importance in the Ethereum ecosystem (L1) or within a parent repository context (L3).

**Request Body:**
```json
{
  "repo_a": "https://github.com/ethereum/go-ethereum",
  "repo_b": "https://github.com/ethereum/solidity", 
  "parent": "ethereum",
  "model": "openai/gpt-4o",
  "temperature": 0.7
}
```

**Request Schema:**
```typescript
interface ComparisonRequest {
  repo_a: string;           // Required: First repository URL
  repo_b: string;           // Required: Second repository URL  
  parent: string;           // Required: "ethereum" for L1, repo URL for L3
  model?: string;           // Optional: Model ID (default: "openai/gpt-4o")
  temperature?: number;     // Optional: 0.0-1.0 (default: 0.7)
}
```

**Response:**
```json
{
  "choice": "A",
  "multiplier": 0.85,
  "raw_uncertainty": 0.23,
  "calibrated_uncertainty": 0.18,
  "explanation": "Repository A (go-ethereum) is the primary Ethereum client implementation...",
  "model_used": "openai/gpt-4o",
  "temperature": 0.7,
  "processing_time_ms": 1250,
  "method": "llm_based",
  "comparison_level": "L1"
}
```

**Response Schema:**
```typescript
interface ComparisonResponse {
  choice: "A" | "B" | "Equal";              // Which repository is preferred
  multiplier: number;                       // Confidence multiplier (0.0-1.0)
  raw_uncertainty: number;                  // Raw uncertainty score
  calibrated_uncertainty: number;           // Calibrated uncertainty score
  explanation: string;                      // Detailed reasoning
  model_used: string;                       // Model that generated response
  temperature: number;                      // Temperature used
  processing_time_ms: number;              // Processing time in milliseconds
  method: "llm_based" | "special_case_criteria"; // Method used
  comparison_level: "L1" | "L3";           // Comparison level
}
```

**Valid Parent Values:**
- `"ethereum"` - L1 comparison (Ethereum ecosystem importance)
- Repository URL - L3 comparison (dependency comparison within parent repo)

### 2. Originality Assessment

**Endpoint:** `POST /assess`

**Purpose:** Assess how original vs dependency-reliant a repository is on a 1-10 scale.

**Request Body:**
```json
{
  "repo": "https://github.com/ethereum/go-ethereum",
  "model": "openai/gpt-4o",
  "temperature": 0.7
}
```

**Request Schema:**
```typescript
interface OriginalityRequest {
  repo: string;             // Required: Repository URL
  model?: string;           // Optional: Model ID
  temperature?: number;     // Optional: 0.0-1.0
}
```

**Response:**
```json
{
  "originality": 8.2,
  "uncertainty": 0.15,
  "explanation": "This repository demonstrates high originality with significant protocol implementation...",
  "criteria_scores": {
    "protocol_implementation": 9.1,
    "algorithmic_innovation": 7.8,
    "developer_experience": 8.5,
    "architectural_innovation": 8.0,
    "security_innovation": 8.7,
    "standards_leadership": 9.2,
    "cross_client_compatibility": 7.5
  },
  "model_used": "openai/gpt-4o",
  "temperature": 0.7,
  "processing_time_ms": 3200,
  "repository_category": "A"
}
```

**Response Schema:**
```typescript
interface OriginalityResponse {
  originality: number;                      // Overall originality score (1-10)
  uncertainty: number;                      // Uncertainty in assessment
  explanation: string;                      // Detailed reasoning
  criteria_scores: {                        // Scores for each criterion
    protocol_implementation: number;
    algorithmic_innovation: number;
    developer_experience: number;
    architectural_innovation: number;
    security_innovation: number;
    standards_leadership: number;
    cross_client_compatibility: number;
  };
  model_used: string;                       // Model identifier
  temperature: number;                      // Temperature used
  processing_time_ms: number;               // Processing time
  repository_category: "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I";
}
```

### 3. Criteria Assessment

**Endpoint:** `POST /criteria`

**Purpose:** Evaluate repository against 11 detailed importance criteria for the Ethereum ecosystem.

**Request Body:**
```json
{
  "repo": "https://github.com/ethereum/go-ethereum",
  "model": "openai/gpt-4o",
  "temperature": 0.7,
  "criteria_weights": {
    "ecosystem_maturity": 0.12,
    "developer_adoption": 0.11,
    "innovation_factor": 0.10
  }
}
```

**Request Schema:**
```typescript
interface CriteriaRequest {
  repo: string;                             // Required: Repository URL
  model?: string;                           // Optional: Model ID  
  temperature?: number;                     // Optional: Temperature
  criteria_weights?: {                      // Optional: Custom criteria weights
    [criterion: string]: number;
  };
}
```

**Response:**
```json
{
  "target_score": 8.45,
  "criteria_scores": {
    "ecosystem_maturity": {
      "score": 9.2,
      "weight": 0.12,
      "reasoning": "Highly mature with extensive ecosystem support..."
    },
    "developer_adoption": {
      "score": 8.8,
      "weight": 0.11, 
      "reasoning": "Widely adopted by developers..."
    }
  },
  "overall_reasoning": "This repository demonstrates exceptional importance...",
  "uncertainty": 0.08,
  "model_used": "openai/gpt-4o",
  "processing_time_ms": 2800
}
```

---

## Status and Debug Endpoints

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "llm-repository-api", 
  "version": "1.1.0",
  "special_case": {
    "enabled": true,
    "data_valid": true,
    "available_repositories": 25
  }
}
```

### API Information

**Endpoint:** `GET /`

**Response:**
```json
{
  "service": "LLM Repository Assessment API",
  "version": "1.1.0",
  "features": [
    "Repository comparison with special case support",
    "Originality assessment",
    "Criteria-based evaluation", 
    "Automatic fallback to LLM when needed"
  ],
  "endpoints": {
    "health": "/health",
    "comparison": "/compare",
    "originality": "/assess", 
    "criteria": "/criteria",
    "special_case_status": "/special-case/status"
  }
}
```

### Special Case Status

**Endpoint:** `GET /special-case/status`

**Response:**
```json
{
  "special_case": {
    "enabled": true,
    "data_valid": true,
    "available_repositories": 25,
    "total_assessments": 127
  },
  "validation": {
    "valid": true,
    "errors": [],
    "total_assessments": 127
  },
  "description": {
    "enabled": "Special case uses criteria assessment data for L1 comparisons",
    "fallback": "Automatically falls back to LLM when assessment data unavailable",
    "data_source": "data/processed/criteria_assessment/detailed_assessments.json"
  }
}
```

### Comparison Methods Debug

**Endpoint:** `GET /debug/comparison-methods`

**Response:**
```json
{
  "comparison_methods": {
    "special_case_criteria": {
      "description": "Uses pre-computed criteria assessment scores",
      "conditions": [
        "parent == 'ethereum'",
        "Both repositories have assessment data available",
        "Special case handling is enabled"
      ],
      "advantages": [
        "Fast response (no LLM call)",
        "Consistent scoring", 
        "Detailed reasoning based on criteria"
      ]
    },
    "llm_based": {
      "description": "Uses Language Model for comparison",
      "conditions": [
        "Special case conditions not met",
        "Fallback when assessment data unavailable"
      ],
      "models_available": [
        "openai/gpt-4o",
        "meta-llama/llama-4-maverick",
        "x-ai/grok-3-beta",
        "deepseek/deepseek-chat-v3-0324"
      ]
    }
  }
}
```

---

## Supported Models

The API supports 12+ LLM models via OpenRouter:

### Primary Models
- `openai/gpt-4o` (default)
- `meta-llama/llama-4-maverick`
- `deepseek/deepseek-chat`
- `anthropic/claude-3-sonnet`

### Secondary Models  
- `x-ai/grok-3-beta`
- `deepseek/deepseek-chat-v3-0324`
- `google/gemini-pro`
- `cohere/command-r-plus`

---

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": "Validation error",
  "detail": "Repository URL is required"
}
```

**HTTP Status Codes:**
- `200` - Success
- `400` - Bad Request (validation error)
- `500` - Internal Server Error
- `503` - Service Unavailable (not initialized)

---

## Repository URL Format

All repository URLs must be valid GitHub URLs:
```
https://github.com/{owner}/{repo}
```

**Examples:**
- `https://github.com/ethereum/go-ethereum` ✓
- `https://github.com/ethereum/solidity` ✓ 
- `github.com/ethereum/go-ethereum` ✗ (missing https://)
- `https://gitlab.com/project/repo` ✗ (not GitHub)

---

## Rate Limiting

- Default: 60 requests per minute
- Configurable via `RATE_LIMIT_RPM` environment variable
- Rate limiting applied per model via OpenRouter

---

## Example Usage

### Basic Repository Comparison
```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "repo_a": "https://github.com/ethereum/go-ethereum",
    "repo_b": "https://github.com/ethereum/solidity",
    "parent": "ethereum"
  }'
```

### Originality Assessment
```bash
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "https://github.com/ethereum/go-ethereum",
    "model": "openai/gpt-4o"
  }'
```

### Health Check
```bash
curl http://localhost:8000/health
```

---

## Data Levels

The API handles three assessment levels:

### Level 1 (L1): Ethereum Ecosystem Comparisons
- **Trigger:** `parent == "ethereum"`
- **Purpose:** Compare repository importance to Ethereum ecosystem
- **Method:** Special case (criteria-based) or LLM fallback
- **Output:** A vs B vs Equal preference

### Level 2 (L2): Originality Assessment  
- **Endpoint:** `/assess`
- **Purpose:** Evaluate repository originality (1-10 scale)
- **Method:** LLM-based assessment with 8 criteria
- **Categories:** A-I repository classification

### Level 3 (L3): Dependency Comparisons
- **Trigger:** `parent == <repository_url>`
- **Purpose:** Compare dependencies within parent repository  
- **Method:** LLM-based comparison
- **Output:** A vs B vs Equal preference

---

## Special Case Handling

For L1 comparisons (`parent == "ethereum"`), the API can use pre-computed criteria assessment scores for faster, more consistent results:

### When Special Case is Used:
1. Both repositories have assessment data available
2. Special case handling is enabled
3. Parent is "ethereum"

### Benefits:
- Fast response (no LLM call required)
- Consistent scoring across requests
- Detailed reasoning based on criteria scores

### Fallback:
- Automatically falls back to LLM-based comparison when special case conditions aren't met
- Transparent to the client (same response format)