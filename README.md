# Åboat Object Tracking and Collision Avoidance

The **Åboat** project, undertaken as part of the 2024-2025 project course for first-year ÅA graduate students. This project focuses on developing navigation and collision avoidance systems for a semi-autonomous vessel designed to operate in maritime environments, semi-autonomously.

## Objectives
- To design and implement a reliable collision avoidance system utilising sensor fusion, including **LIDAR** and multiple cameras for a 360-degree spatial awareness.
- To synchronise real-time image and LIDAR data for accurate detection and classification of obstacles, such as other vessels, rocks, and shoreline features.
- To ensure seamless integration with waypoint-following capabilities, for navigation.
- To develop and refine algorithms that enable the Åboat to make autonomous decisions for safe and efficient navigation in dynamic environments (under supervision).

## (Planned) Features
1. **Real-Time Obstacle Detection**:
   - Utilising a combination of LIDAR sensors and camera feeds to identify potential collisions and objects in the water.
   - Processing data in real-time to detect, classify, and map obstacles.

2. **Autonomous Decision-Making**:
   - Algorithms capable of dynamically rerouting the vessel to avoid obstacles and maintain the planned course.
   - Fail-safe mechanisms to minimise risks in unexpected scenarios.

3. **Simulator Integration**:
   - Testing and refining the system in a virtual maritime environment using the **AILiveSim** simulator integrated with Unreal Engine 4.
   - Simulating real-world scenarios to validate system performance before physical deployment and evaluation.
