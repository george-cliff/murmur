
-- TABLES

-- NOTE: all timestampes ISO-8601
-- NOTE: dont forget PRAGMA foreign_keys = ON; when wiring up the tables into code
-- mutable tables get created_at/updated_at/deleted_at (rows can be edited or soft-deleted later).
--  write-once tables (unit_heartbeats, unit_faults, moon_phases, daily_weather) only get created_at, because a row here never changes after it's inserted.



CREATE TABLE IF NOT EXISTS units (
    id TEXT PRIMARY KEY, --ULID
    serial_number TEXT NOT NULL, --e.g. MM-STAR-20XX-XXXX
    display_name TEXT, -- optinal user-generated nickname, e.g. "Pine Tree Starling"
    current_model_version_id INTEGER REFERENCES model_versions(id) ON DELETE RESTRICT, -- which model version this unit is running, RESTRICT: blocks deleting this model version while any unit still points at it
    created_at TEXT DEFAULT CURRENT_TIMESTAMP, -- when this row was first inserted
    updated_at TEXT, -- when this row was last modified (NULL until the first update)
    deleted_at TEXT -- when this row was "deleted" rows kept for history, NULL when row is live
);

CREATE TABLE IF NOT EXISTS deployments (
    id INTEGER PRIMARY KEY,
    unit_id TEXT REFERENCES units(id) ON DELETE RESTRICT NOT NULL,
    site_id INTEGER REFERENCES sites(id) ON DELETE RESTRICT,
    tenant_id INTEGER REFERENCES tenants(id) ON DELETE RESTRICT,
    started_at TEXT NOT NULL, -- when deployment started
    ended_at TEXT, -- when deployment finihsed, null means deplyoment is still live
    latitude REAL NOT NULL, -- specific coords for the starling unit
    longitude REAL NOT NULL, -- specific coords for the starling unit
    elevation_m REAL, -- metres above sea level at this deployment's coords, e.g. 12.0
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    deleted_at TEXT
);


-- GDPR: this is the one table in the schema holding personal data, isolated here for easier auditing.
-- deleted_at is for when all personal information: name, phone_number, email, address, notes are nulled, retention_review_at also gets nulled
-- lawful_basis, consent_given_at, erasure_requested_at, created_at, deleted_at get kept in the soft delete.
CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY,
    name TEXT, -- e.g. "Full Name"
    phone_number TEXT, -- e.g +447123 456789
    email TEXT, -- e.g. "name@example.com"
    address TEXT, -- e.g. "123 High Street, Townname, AB1 2CD"
    notes TEXT, -- free text, any additional context about this contact
    lawful_basis TEXT CHECK (lawful_basis IN ('consent', 'contract', 'legal_obligation', 'vital_interests', 'public_task', 'legitimate_interests')), -- GDPR stuff
    consent_given_at TEXT, -- if lawful_basis = consent when they agreed to this
    erasure_requested_at TEXT, -- set when a request comes in but before it's actually processed, allows for auditable verification rather than deleting instantly
    retention_review_at TEXT, -- next date to check "do we still need this contact" (GDPR storage limitation principle)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS sites (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL, --  e.g. "Riverside Park"
    tenant_id INTEGER REFERENCES tenants(id) ON DELETE RESTRICT,
    latitude REAL, -- general coords for the site (e.g. coords for site office)
    longitude REAL, -- general coords for the site (e.g. coords for  site office)
    description TEXT, -- free text, e.g. "Community green space with mature oak trees, open to public 08:00-21:00"
    contact_id INTEGER REFERENCES contacts(id) ON DELETE SET NULL, -- so a contact can be deleted without blocking this site
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS tenants (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL, -- e.g. "Parish Council"
    contact_id INTEGER REFERENCES contacts(id) ON DELETE SET NULL, -- the tenant's main contact, deletable without blocking this tenant
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS current_weather (
    id INTEGER PRIMARY KEY,
    unit_id TEXT REFERENCES units(id) ON DELETE RESTRICT NOT NULL,
    recorded_at TEXT NOT NULL, -- when this reading was taken
    temperature_c REAL, -- degrees Celsius, e.g. 4.2
    apparent_temperature_c REAL, -- "feels like" temperature, e.g. 3.1
    humidity_pct REAL, -- relative humidity, 0-100, e.g. 88
    pressure_hpa REAL, -- mean sea-level pressure, hectopascals, e.g. 1013.25
    surface_pressure_hpa REAL, -- raw station pressure, not adjusted for elevation
    wind_speed_kmh REAL, -- km/h, e.g. 12.4
    wind_gusts_kmh REAL, -- km/h, e.g. 22.0
    wind_direction_deg INTEGER, -- 0-360, e.g. 278
    precipitation_mm REAL, -- mm, e.g. 0.2
    rain_mm REAL, -- mm, e.g. 0.2
    showers_mm REAL, -- mm, nullable (API returns null outside shower conditions)
    snowfall_cm REAL, -- cm, e.g. 0.0
    cloud_cover_pct REAL, -- 0-100, e.g. 100
    weather_code INTEGER, -- WMO weather code, e.g. 3
    is_day BOOLEAN, -- 1 if daylight at recorded_at, 0 if night
    source TEXT NOT NULL CHECK (source IN ('api', 'sensor')), -- for future on device sensors
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS daily_weather (
    id INTEGER PRIMARY KEY,
    lat_bucket INTEGER NOT NULL, -- degrees * 100 (centidegrees) gives an ~1.1km grid, so nearby Starling devices at the same site naturally share one bucket -- INTEGER not REAL so UNIQUE's exact-match comparison is reliable, e.g. 5220 for 52.20 degrees
    lon_bucket INTEGER NOT NULL, -- see above
    observation_date TEXT NOT NULL, -- e.g. "2026-07-06"
    sunrise_time TEXT, -- e.g. "2026-07-06T04:47"
    sunset_time TEXT, -- e.g. "2026-07-06T21:21"
    temperature_max_c REAL, -- e.g. 31.3
    temperature_min_c REAL, -- e.g. 16.7
    apparent_temperature_max_c REAL, -- e.g. 29.2
    apparent_temperature_min_c REAL, -- e.g. 16.8
    precipitation_sum_mm REAL, -- e.g. 0.0
    rain_sum_mm REAL, -- e.g. 0.0
    showers_sum_mm REAL, -- mm, nullable
    snowfall_sum_cm REAL, -- e.g. 0.0
    precipitation_hours REAL, -- e.g. 0.0
    precipitation_probability_max_pct REAL, -- 0-100, nullable
    wind_speed_max_kmh REAL, -- km/h, e.g. 28.1
    wind_gusts_max_kmh REAL, -- km/h, e.g. 51.5
    wind_direction_dominant_deg INTEGER, -- 0-360, e.g. 260
    shortwave_radiation_sum_mjm2 REAL, -- MJ/m², e.g. 27.52
    et0_fao_evapotranspiration_mm REAL, -- mm, e.g. 7.4
    daylight_duration_s REAL, -- seconds, e.g. 59670.12
    sunshine_duration_s REAL, -- seconds, e.g. 47550.38
    uv_index_max REAL, -- e.g. 5.65
    uv_index_clear_sky_max REAL, -- nullable
    weather_code INTEGER, -- WMO weather code, e.g. 3
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (lat_bucket, lon_bucket, observation_date)
);

CREATE TABLE IF NOT EXISTS current_air_quality (
    id INTEGER PRIMARY KEY,
    unit_id TEXT REFERENCES units(id) ON DELETE RESTRICT NOT NULL,
    recorded_at TEXT NOT NULL, -- when this reading was taken
    european_aqi INTEGER, -- EAQI index, e.g. 17
    us_aqi INTEGER, -- USAQI index, e.g. 22
    pm10_ugm3 REAL, -- μg/m³, e.g. 6.9
    pm2_5_ugm3 REAL, -- μg/m³, e.g. 4.9
    carbon_monoxide_ugm3 REAL, -- μg/m³, e.g. 122.0
    nitrogen_dioxide_ugm3 REAL, -- μg/m³, e.g. 8.6
    sulphur_dioxide_ugm3 REAL, -- μg/m³, e.g. 0.7
    ozone_ugm3 REAL, -- μg/m³, e.g. 43.0
    aerosol_optical_depth REAL, -- unitless, e.g. 0.20
    dust_ugm3 REAL, -- μg/m³, e.g. 0.0
    uv_index REAL, -- unitless, e.g. 0.10
    uv_index_clear_sky REAL, -- unitless, e.g. 0.20
    ammonia_ugm3 REAL, -- μg/m³, e.g. 2.0
    alder_pollen_grainsm3 REAL, -- grains/m³, e.g. 0.0
    birch_pollen_grainsm3 REAL, -- grains/m³, e.g. 0.0
    mugwort_pollen_grainsm3 REAL, -- grains/m³, e.g. 0.3
    grass_pollen_grainsm3 REAL, -- grains/m³, e.g. 0.7
    olive_pollen_grainsm3 REAL, -- grains/m³, e.g. 0.0
    ragweed_pollen_grainsm3 REAL, -- grains/m³, e.g. 0.0
    source TEXT NOT NULL CHECK (source IN ('api', 'sensor')), -- for future on device sensors
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS moon_phases (
    observation_date TEXT PRIMARY KEY, -- "2026-07-06"
    phase TEXT NOT NULL, -- e.g. "Waxing Gibbous"
    illumination_pct REAL, -- 0-100, e.g. 78.4
    age_days REAL, -- days since the last new moon, 0 to ~29.5
    distance_km REAL, -- Earth-Moon distance right now, e.g. 384400.0 (closer moon = physically brighter for the same phase)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY,
    version_label TEXT NOT NULL, -- e.g "perch-v2" or "sparrowhawk-V9Q"
    label_list_reference TEXT NOT NULL, -- which labels.csv snapshot this version shipped with, e.g. "labels_v2.csv"
    released_at TEXT, -- when this model version was released e.g. "2026-07-06T10:00:00"
    notes TEXT, -- free text, e.g. what changed from the previous version
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS model_labels (
    id INTEGER PRIMARY KEY,
    model_version_id INTEGER REFERENCES model_versions(id) ON DELETE RESTRICT NOT NULL,
    class_index INTEGER NOT NULL, -- Perch's output index for this class, e.g. 101
    raw_label_name TEXT NOT NULL, -- literal string from labels.csv, e.g. "Sturnus vulgaris" or "Aircraft" (non-taxon)
    is_taxon BOOLEAN NOT NULL, -- whether this class actually resolves to a species at all, e.g. false for non-taxon classes like "Aircraft"
    taxon_id INTEGER REFERENCES taxa(id) ON DELETE RESTRICT, -- nullable: either is_taxon is false (not a species at all), or it is a species but has no matching row in taxa yet (which should be flagged)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    deleted_at TEXT,
    UNIQUE (model_version_id, class_index)
);


CREATE TABLE IF NOT EXISTS taxa (
    id INTEGER PRIMARY KEY,
    scientific_name TEXT NOT NULL, --  e.g. "Sturnus vulgaris"
    common_name_en TEXT, -- e.g. "Common Starling"
    taxon_group TEXT NOT NULL CHECK (taxon_group IN ('Aves', 'Amphibia', 'Insecta', 'Mammalia', 'Reptilia')), -- one of the 5 taxonomic groups Perch spans, e.g. "Aves" (should all be Aves to begin with)
    ebird_id TEXT, -- eBird species code, e.g. "eursta"
    gbif_id TEXT, -- GBIF taxon key, e.g. "9809229"
    inaturalist_id TEXT, -- iNaturalist taxon ID, e.g. "14850"
    avibase_id TEXT, -- Avibase ID, e.g. "94A4403295E2D9BE"
    birdlife_id TEXT, -- BirdLife International ID, e.g. "22710886"
    is_sensitive BOOLEAN NOT NULL DEFAULT 0, -- for rare/protected species needing location redaction
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS taxon_aliases (
    id INTEGER PRIMARY KEY,
    taxon_id INTEGER REFERENCES taxa(id) ON DELETE CASCADE NOT NULL, -- if parent is delted remove aliases too
    alias_scientific_name TEXT NOT NULL, -- e.g. "Sturnus vulgaris Linnaeus"
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY,
    unit_id TEXT REFERENCES units(id) ON DELETE RESTRICT NOT NULL,
    deployment_id INTEGER REFERENCES deployments(id) ON DELETE RESTRICT NOT NULL,
    sequence_no INTEGER NOT NULL, -- for de-duplicating re-sent transmissions
    detected_at TEXT NOT NULL, -- when this actually happened on Starling
    received_at TEXT, -- when Nest actually received
    model_version_id INTEGER REFERENCES model_versions(id) ON DELETE RESTRICT NOT NULL,
    model_label_id INTEGER REFERENCES model_labels(id) ON DELETE RESTRICT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1), -- post-sigmoid probability, e.g. 0.94
    current_weather_id INTEGER REFERENCES current_weather(id) ON DELETE SET NULL,
    daily_weather_id INTEGER REFERENCES daily_weather(id) ON DELETE SET NULL,
    current_air_quality_id INTEGER REFERENCES current_air_quality(id) ON DELETE SET NULL,
    moon_phase_date TEXT REFERENCES moon_phases(observation_date) ON DELETE SET NULL,
    local_audio_path TEXT, -- path on Starling's own storage (SD card/SSD) right after recording e.g. "/media/sd/clips/2026-07-06T06-05-00.wav"
    nest_audio_path TEXT, -- populated once the WAV is actually transferred to Nest's storage e.g. "/srv/nest/audio/MM-STAR-20.../2026-07-06T06-05-00.wav"
    file_transferred_at TEXT, -- when the transfer from local to nest actually happened
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    deleted_at TEXT,
    UNIQUE (unit_id, sequence_no)
);

CREATE TABLE IF NOT EXISTS unit_heartbeats (
    id INTEGER PRIMARY KEY,
    unit_id TEXT REFERENCES units(id) ON DELETE RESTRICT NOT NULL,
    recorded_at TEXT NOT NULL, -- when this heartbeat was sent
    received_at TEXT, -- when Nest actually received
    battery_pct REAL, -- 0-100, e.g. 87.5
    reported_model_version_id INTEGER REFERENCES model_versions(id) ON DELETE RESTRICT, -- what the unit itself claims it's running, vs units.current_model_version_id (what it *should* be running)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS unit_faults (
    id INTEGER PRIMARY KEY,
    unit_id TEXT REFERENCES units(id) ON DELETE RESTRICT NOT NULL,
    occurred_at TEXT NOT NULL, -- when this fault happened
    received_at TEXT, -- when Nest actually received
    level TEXT NOT NULL CHECK (level IN ('emergency', 'alert', 'critical', 'error', 'warning', 'notice', 'informational', 'debug')), -- one of the 8 RFC 5424 syslog levels (see docs/logging-standards.md), e.g. "warning"
    event_code TEXT, -- short code for multi use, e.g. "BATT_LOW"
    message TEXT, -- human-readable detail, e.g. "Battery below 20%" "Battery below 10%"
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes

-- limits a single unit to one open deployment (ended_at IS NULL) at a time
CREATE UNIQUE INDEX IF NOT EXISTS idx_deployments_one_open_per_unit
ON deployments (unit_id)
WHERE ended_at IS NULL AND deleted_at IS NULL;

-- finds the closest weather reading to a detection's timestamp, for a given unit
CREATE INDEX IF NOT EXISTS idx_current_weather_unit_time
ON current_weather (unit_id, recorded_at);

-- finds the closest air quality reading to a detection's timestamp, for a given unit
CREATE INDEX IF NOT EXISTS idx_current_air_quality_unit_time
ON current_air_quality (unit_id, recorded_at);

-- groups detections by species, e.g. detection count per species
CREATE INDEX IF NOT EXISTS idx_model_labels_taxon
ON model_labels (taxon_id);

-- finds detections in a time range across the whole fleet, ignoring unit_id
CREATE INDEX IF NOT EXISTS idx_detections_detected_at
ON detections (detected_at);

-- finds detections from one unit within a time window
CREATE INDEX IF NOT EXISTS idx_detections_unit_time
ON detections (unit_id, detected_at);

-- finds one unit's heartbeat history, in order
CREATE INDEX IF NOT EXISTS idx_unit_heartbeats_unit_time
ON unit_heartbeats (unit_id, recorded_at);

-- finds one unit's fault history, in order
CREATE INDEX IF NOT EXISTS idx_unit_faults_unit_time
ON unit_faults (unit_id, occurred_at);

-- lets a decommissioned unit's serial number be reused
CREATE UNIQUE INDEX IF NOT EXISTS idx_units_serial_number_live
ON units (serial_number)
WHERE deleted_at IS NULL;

-- lets a closed site's name be reused
CREATE UNIQUE INDEX IF NOT EXISTS idx_sites_name_live
ON sites (name)
WHERE deleted_at IS NULL;

-- lets an ended tenant's name be reused
CREATE UNIQUE INDEX IF NOT EXISTS idx_tenants_name_live
ON tenants (name)
WHERE deleted_at IS NULL;

-- lets a retired model version's label be reused
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_version_label_live
ON model_versions (version_label)
WHERE deleted_at IS NULL;

-- lets a removed taxon's scientific name be reused, e.g. after a bad import
CREATE UNIQUE INDEX IF NOT EXISTS idx_taxa_scientific_name_live
ON taxa (scientific_name)
WHERE deleted_at IS NULL;

-- lets a removed alias's name be reused, e.g. re-attached to a different taxon
CREATE UNIQUE INDEX IF NOT EXISTS idx_taxon_aliases_alias_scientific_name_live
ON taxon_aliases (alias_scientific_name)
WHERE deleted_at IS NULL;

-- joins a detection back to its matched species/class
CREATE INDEX IF NOT EXISTS idx_detections_model_label ON detections (model_label_id);

-- joins a detection back to the model version that produced it
CREATE INDEX IF NOT EXISTS idx_detections_model_version ON detections (model_version_id);

-- joins a detection back to its deployment
CREATE INDEX IF NOT EXISTS idx_detections_deployment ON detections (deployment_id);

-- joins a detection back to that day's moon phase
CREATE INDEX IF NOT EXISTS idx_detections_moon_phase ON detections (moon_phase_date);

-- joins a deployment back to its site
CREATE INDEX IF NOT EXISTS idx_deployments_site ON deployments (site_id);

-- joins a deployment back to its tenant
CREATE INDEX IF NOT EXISTS idx_deployments_tenant ON deployments (tenant_id);

-- joins a site back to its tenant
CREATE INDEX IF NOT EXISTS idx_sites_tenant ON sites (tenant_id);

-- joins a site back to its contact
CREATE INDEX IF NOT EXISTS idx_sites_contact ON sites (contact_id);

-- joins a tenant back to its contact
CREATE INDEX IF NOT EXISTS idx_tenants_contact ON tenants (contact_id);
