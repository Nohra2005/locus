# 📍 Locus: Intelligent Visual Search Engine

> **Bridging the gap between digital inspiration and local availability.**

**Locus** is a domain-specific visual search engine designed to retrieve fashion inventory based on visual similarity. It addresses the "Inspiration-Availability Gap" in Lebanon by allowing users to upload photos of clothing and instantly find the closest matching items available in local physical stores.

---

## 📖 Table of Contents
- [Problem Statement](#-problem-statement)
- [Project Objective](#-project-objective)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technical Constraints](#-technical-constraints)
- [Success Metrics](#-success-metrics)
- [Roadmap & Future Work](#-roadmap--future-work)

---

## 🚩 Problem Statement

### The Context
Online shopping dominates the global fashion industry, but the experience remains fragmented for consumers in Lebanon.
1.  **Logistical Barriers:** International shipping is often slow, expensive, or unavailable.
2.  **The "Try-On" Risk:** Users cannot verify fit, fabric quality, or sizing before purchase.
3.  **Inefficient Physical Retail:** While local malls possess the inventory, finding a specific item seen on social media (Pinterest/Instagram) requires hours of fruitless physical searching.

### The Impact
Shoppers are trapped between "unobtainable inspiration" online and "unsearchable inventory" locally. Locus merges the convenience of online search with the instant gratification and reliability of physical retail.

---

## 🎯 Project Objective

To design and develop a **Hybrid Visual Recommendation System** that ranks inventory by **Visual Similarity** and **Geographic Nearness**. The system prioritizes **Accuracy over Speed**, ensuring that users find the exact "look" they want within a drivable distance.

---

## ⚡ Key Features

### 1. Core Visual Pipeline
* **Automatic Pre-Processing:** Utilizes `rembg` (U2Net) to remove background noise and isolate the garment.
* **Smart ROI Normalization:** Automatically crops the image to the object's bounding box to improve embedding accuracy.
* **Multi-View Indexing (The "Digital Twin"):** Products are indexed as "folders" containing multiple angles (Front, Back, Lifestyle). If a user uploads a photo of the *back* of a shirt, Locus matches it to the "Back View" vector of the product rather than failing on the "Front View."

### 2. Algorithmic Capabilities
* **Hybrid Retrieval Engine:** Combines **Vector Similarity** (via CLIP embeddings) with **Hard Filters** (Category, Location, Price).
* **Smart Categorization:** Automatically classifies inputs (e.g., "Dress", "Coat") to narrow the search space.
* **Adversarial Robustness:**
    * **Ghost Check:** Rejects empty images where background removal failed.
    * **Confidence Thresholding:** Filters out predictions with <45% confidence to prevent "hallucinated" matches.
    * **Negative Anchors:** Explicitly rejects non-fashion objects (e.g., cars, animals, food).

### 3. Recommendation & Personalization
* **"User Persona" History (Stochastic Sampling):** Instead of averaging user tastes into a generic mix, the system cycles through the user's interaction history. If a user likes *Leather Jackets* and *Summer Dresses*, the home feed will dynamically alternate between recommending these distinct styles.
* **"Vibe-Check" Outfit Completion:** Uses **Zero-Shot Style Anchoring** to classify the aesthetic of a search (e.g., "Bohemian", "Minimalist") and suggests accessories that strictly match that visual vibe.

### 4. User Interface
* **Visual Dashboard:** Responsive web app with a "Vision Debugger" showing users exactly what the AI "sees."
* **Smart Crop Tool:** Manual override for users to adjust the focus area.
* **Local Availability Map:** Visually displays the nearest store stocking the matched item.
* **Shop Owner Portal:** A dedicated dashboard for retailers to bulk-upload inventory and manage stock levels.

---

## 🏗 System Architecture

The system follows a microservices architecture running on Docker:

| Service | Technology | Purpose |
| :--- | :--- | :--- |
| **Gateway** | FastAPI | Orchestrates requests, handles uploads, and manages user auth. |
| **Visual Engine** | PyTorch / CLIP | Handles image processing, embedding generation, and classification. |
| **Vector DB** | Qdrant | Stores high-dimensional embeddings for fast similarity search. |
| **Dashboard** | Streamlit / React | Frontend interface for Users and Shop Owners. |

---

## ⚠️ Technical Constraints & Scope

### In-Scope
* **Platform:** Web-based application (PWA optimized).
* **Hardware:** Optimized for standard CPU architecture (No GPU dependency for inference).
* **Latency:** End-to-end processing allowed up to **15 seconds** (Prioritizing accuracy over real-time speed).

### Out-of-Scope
* **Multi-Object Detection:** The system assumes one primary item per search (no simultaneous Hat + Shoe detection).
* **Mobile App:** Native iOS/Android development is excluded.
* **Payment Gateways:** Transactions happen physically in-store; Locus is a discovery tool.

---

## 📊 Success Metrics (KPIs)

1.  **Accuracy:** In a set of 10 results, at least 8 must be visually relevant to the query.
2.  **Robustness:** 100% rejection rate for low-confidence inputs (<45%) or non-fashion objects.
3.  **Speed:** Complete search pipeline (Upload -> Crop -> Embed -> Rank) must complete in under **15s**.

---

## 🛣 Roadmap & Future Work

* **Human-in-the-Loop Feedback:** Implementing a mechanism where users can correct the AI's classification (e.g., "This is a skirt, not a dress"). These corrections will be stored to fine-tune the model in future iterations.
* **Real-Time Inventory Sync:** API integration with retailer POS systems to update stock levels automatically, replacing the manual Shop Owner upload.
* **AR Virtual Try-On:** Overlaying retrieved items onto the user's camera feed using generative AI.

---

## 🛠 Setup & Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/locus.git](https://github.com/your-username/locus.git)

# Navigate to directory
cd locus

# Start the services
docker-compose up --build
