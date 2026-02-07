# Project Scope Statement: Locus Visual Search Engine

## 1. Problem Statement (The "Why")
**Current Issue:**
Online shopping has become a major part of today’s fashion industry. The main inconvenience is clients buy without trying/seeing the product and have to wait for shipping. While many shoppers really enjoy going out to the mall, physical shopping is unarguably draining, inefficient and often unfruitful: shoppers can spend countless hours looking for a specific product they have in mind or they have found on Pinterest/Instagram but have no idea where to find it. Some countries like Lebanon lack a centralized marketplace where people can easily order from a wide variety of products. This emphasizes the need for Lebanese shoppers to physically go out to malls.

**Impact:**
Locus merges best of both worlds: convenience and precision of online shopping with the product experience of physical shopping.

## 2. Project Objective (The "What")
To design and develop a clothing recommendation system that returns similar items ranked by similarity and nearness that allows users to upload an image and retrieve visually similar inventory items with a focus on accuracy and speed.

## 3. In-Scope (Features & Functionalities)

### Core Functionality (The "What")
* **Visual Search Pipeline:** End-to-end processing of user uploads, including automatic background removal, object isolation (ROI Normalization), and vectorization.
* **Multi-View Product Indexing:** Indexes inventory items as multi-vector "folders" (Front, Back, Lifestyle) to ensure accurate retrieval regardless of the input angle.
* **Inventory Localization:** Maps visual search results to specific physical store locations to solve the "shipping delay" pain point.
* **Smart Categorization:** Automatically identifies clothing categories (e.g., "Dress", "Coat") to filter search results.
* **User Authentication:** To keep track of search/purchase history for recommendation engines.

### Algorithmic Capabilities (The "How")
* **Hybrid Retrieval Engine:** Combines Vector Similarity (Visual Match) with Hard Filters (Category, Location, Price) for high-precision results.
* **Adversarial Input Robustness:**
    * Rejects empty/ghost images where background removal failed.
    * Filters out predictions with <45% confidence.

### Recommendation & Personalization Engines
* **"User Persona" History Recommendations:**
    * **Concept:** Builds a dynamic "Taste Profile" for each user based on their interaction history.
    * **Mechanism:** The system cycles through items in a user's history to generate recommendations (Stochastic Sampling).
* **"Vibe-Check" Outfit Completion:**
    * **Concept:** Suggests complementary items (e.g., accessories) that match the aesthetic of the current search.
    * **Mechanism:** Uses Zero-Shot Style Anchoring (via CLIP) to classify the search item's style (e.g., "Bohemian", "Minimalist"). The system then queries the inventory for complementary categories (e.g., Shoes) that share that specific style tag, ensuring a coherent outfit suggestion.

### User Interface
* **Visual Dashboard:** A responsive web interface featuring:
    * **Smart Crop Tool:** Allows users to manually adjust the focus area.
    * **AI Vision Debugger:** Transparent view of the background-removed input.
    * **Local Availability Map:** Displays store locations for matched items.
    * **Recommendation Engine:** Recommends based on search history.
* **Dashboard for Shop Owners:** To upload their inventory.

## 4. Out-of-Scope (The "No-Go" Zone)
* Multi-object detection (detecting a hat and shoes simultaneously).
* Mobile app development (Web only).
* Integration with live payment gateways.
* A system that can 100% detect the category of random inserted objects and affirm with certainty that it is not a piece of clothing (not crucial).

## 5. Technical Constraints & Requirements
* **Performance:** Search latency can be compromised but to a certain extent. The model should be able to find similar looking items and especially “exact match”. We need more accuracy than latency.
* **Infrastructure:** Gateway is for uploading the image, visual engines prepare the image and detect the category, and the ranking search through the quadrant database for the best match.
* **Data:** Retailers catalogue.
* **Hardware:** Must run on standard CPU architecture (no GPU requirement).

## 6. Success Metrics (KPIs)
* **Accuracy:** System correctly categorize items in precise groups and return accurate similar items. At least 8 pictures over 10 are similar to what I am looking for.
* **Robustness:** System successfully rejects low confidence predictions (with 45% resemblance and less).
* **Speed:** End-to-end processing must take maximum 15s.

## 7. Assumptions & Risks
* **Assumption:** User photos will have reasonable lighting and resolution.
* **Risk:** Background removal might fail on white-on-white images.
* **Mitigation:** Implemented alpha-channel check to detect "ghost" images.

---
**Questions & Notes:**
* **Feedback Loop:** When the system fails to detect category or wrongly classifies, user can correct the ML, which will be used for feedback.
* **Inventory Tracking:** How will we be able to keep track of inventory? (To be determined via shop owner dashboard integration).
