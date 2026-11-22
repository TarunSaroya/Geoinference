from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetaDefinition:
    name: str
    category: str
    description: str
    status: str
    priority: str


META_CATALOG = [
    # Road
    MetaDefinition(
        name="driving_side",
        category="road",
        description="Left-hand or right-hand traffic orientation.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="road_surface_characteristics",
        category="road",
        description=(
            "Brightness, saturation, and texture characteristics "
            "of the lower image region."
        ),
        status="DERIVABLE",
        priority="MEDIUM",
    ),
    MetaDefinition(
        name="lane_marking_candidates",
        category="road",
        description=(
            "Potential painted road markings detected using "
            "bright-region and shape analysis."
        ),
        status="DERIVABLE",
        priority="HIGH",
    ),
    MetaDefinition(
        name="lane_marking_distribution",
        category="road",
        description=(
            "Spatial distribution of potential lane-marking "
            "candidates across the image."
        ),
        status="DERIVABLE",
        priority="HIGH",
    ),
    MetaDefinition(
        name="road_width",
        category="road",
        description="Estimated road width or road-scale geometry.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    MetaDefinition(
        name="road_geometry",
        category="road",
        description="Road direction, perspective, curvature, and geometric structure.",
        status="DERIVABLE",
        priority="MEDIUM",
    ),
    # Infrastructure
    MetaDefinition(
        name="overhead_wire_candidates",
        category="infrastructure",
        description=(
            "Potential overhead wiring segments detected through "
            "edge and line geometry analysis."
        ),
        status="DERIVABLE",
        priority="HIGH",
    ),
    MetaDefinition(
        name="utility_pole_candidates",
        category="infrastructure",
        description=("Individual pole-like regions detected by semantic segmentation."),
        status="DERIVABLE",
        priority="HIGH",
    ),
    MetaDefinition(
        name="utility_pole_type",
        category="infrastructure",
        description="Shape and structural type of utility poles.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="utility_pole_material",
        category="infrastructure",
        description="Likely concrete, wood, metal, or other pole material.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="wiring_pattern",
        category="infrastructure",
        description="Arrangement and characteristics of overhead wiring.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="transformer_presence",
        category="infrastructure",
        description="Presence and characteristics of electrical transformers.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    MetaDefinition(
        name="streetlight_style",
        category="infrastructure",
        description="Streetlight design and placement.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    # Signs
    MetaDefinition(
        name="traffic_sign_presence",
        category="signs",
        description="Presence and spatial location of traffic signs.",
        status="DERIVABLE",
        priority="HIGH",
    ),
    MetaDefinition(
        name="sign_shape",
        category="signs",
        description="Shape of detected signs.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="sign_color",
        category="signs",
        description="Dominant and structural sign colors.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="sign_layout",
        category="signs",
        description="Visual layout and conventions of detected signs.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="traffic_light_style",
        category="signs",
        description="Traffic-light configuration and visual style.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    # Text
    MetaDefinition(
        name="detected_text",
        category="text",
        description="Text detected in the scene.",
        status="AVAILABLE",
        priority="MEDIUM",
    ),
    MetaDefinition(
        name="language_or_script",
        category="text",
        description="Language or writing system suggested by visible text.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="place_name",
        category="text",
        description="Detected geographic place names.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="road_name",
        category="text",
        description="Detected street or road names.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    # Vehicles
    MetaDefinition(
        name="vehicle_type",
        category="vehicles",
        description="Types of vehicles visible in the scene.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    MetaDefinition(
        name="vehicle_orientation",
        category="vehicles",
        description="Vehicle orientation and possible traffic direction.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="license_plate_shape",
        category="vehicles",
        description="Shape and aspect ratio of visible license plates.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="license_plate_color",
        category="vehicles",
        description="Visible license-plate color characteristics.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    # Architecture
    MetaDefinition(
        name="architecture_style",
        category="architecture",
        description="Regional architectural characteristics.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="roof_type",
        category="architecture",
        description="Roof shape and construction style.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    MetaDefinition(
        name="building_material",
        category="architecture",
        description="Visible facade and construction materials.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    MetaDefinition(
        name="building_density",
        category="architecture",
        description="Density and spatial arrangement of buildings.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    # Environment
    MetaDefinition(
        name="vegetation_type",
        category="environment",
        description="Visible vegetation and broad vegetation type.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="biome_or_climate",
        category="environment",
        description="Broad environmental or climatic characteristics.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="terrain_type",
        category="environment",
        description=(
            "Terrain characteristics such as flat, hilly, mountainous, or rocky."
        ),
        status="DERIVABLE",
        priority="HIGH",
    ),
    MetaDefinition(
        name="mountain_presence",
        category="environment",
        description="Presence and spatial characteristics of mountains.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="mountain_characteristics",
        category="environment",
        description=(
            "Shape, elevation profile, and visual characteristics of mountains."
        ),
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="water_presence",
        category="environment",
        description="Presence of rivers, lakes, sea, or other visible water.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    # Street furniture
    MetaDefinition(
        name="bollards",
        category="street_furniture",
        description="Presence and style of roadside bollards.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="barriers",
        category="street_furniture",
        description="Roadside barriers and their visual characteristics.",
        status="MODEL_REQUIRED",
        priority="HIGH",
    ),
    MetaDefinition(
        name="curbs",
        category="street_furniture",
        description="Curb design and markings.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    MetaDefinition(
        name="guardrails",
        category="street_furniture",
        description="Guardrail presence and style.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    MetaDefinition(
        name="bus_stop_style",
        category="street_furniture",
        description="Bus-stop structures and signage.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    # Camera / imagery
    MetaDefinition(
        name="camera_characteristics",
        category="imagery",
        description="Camera-generation or imagery characteristics.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    MetaDefinition(
        name="image_artifacts",
        category="imagery",
        description="Blur, stitching, compression, panorama, or capture artifacts.",
        status="DERIVABLE",
        priority="MEDIUM",
    ),
    # Geometry / lighting
    MetaDefinition(
        name="shadow_direction",
        category="geometry",
        description="Visible shadow direction and approximate orientation.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    MetaDefinition(
        name="sun_direction",
        category="geometry",
        description="Approximate illumination direction inferred from the scene.",
        status="MODEL_REQUIRED",
        priority="MEDIUM",
    ),
    # Learned geographic prior
    MetaDefinition(
        name="learned_location_prior",
        category="learned_prior",
        description="Independent learned image-to-location prediction.",
        status="AVAILABLE",
        priority="HIGH",
    ),
]
