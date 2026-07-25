# assign_collections.rb — assign Manyfold collections via ActiveRecord
#
# WHY THIS EXISTS
# Manyfold 0.146.0 (and current main) has a bug in the v0 API: setting a model's
# collection via `PATCH /models/{id}` with `isPartOf` raises
#   ArgumentError: missing keyword: :user
# in ManyfoldApi::V0::ModelDeserializer#deserialize — `CollectionDeserializer.new(object: it)`
# omits the required `user:` keyword. So EVERY API collection assignment 500s
# (the pipeline's `upload` collection step and `upload --reconcile-collections`
# both hit this). See bead ModelTagger2-57t.
#
# This script sidesteps the broken deserializer by assigning collections
# directly through ActiveRecord — the same path the web UI uses — so it needs no
# source patch, no restart, and survives upgrades. It mirrors the pipeline's
# reconcile logic: a model's collection is its own "faction:" tag; terrain
# (model_type: Terrain, or a terrain cue word) falls back to a "Terrain"
# collection; models already in a collection are left alone (keep-manual).
# Idempotent.
#
# RUN (from the Manyfold host, in the app container):
#   docker compose exec web bin/rails runner assign_collections.rb                 # DRY RUN
#   LIMIT=50 docker compose exec web bin/rails runner assign_collections.rb        # dry-run, first 50
#   DRY_RUN=false docker compose exec web bin/rails runner assign_collections.rb   # APPLY
#   (or paste the whole file into `bin/rails console`)
#
# VERIFY ONCE against the running source: `keywords_for` reads a model's tags
# (our datapackage "keywords"). `tag_list` is the acts-as-taggable-on accessor;
# if Manyfold uses a different association/context, adjust that one method.

require "set"

DRY_RUN            = ENV.fetch("DRY_RUN", "true") != "false"
LIMIT              = ENV["LIMIT"]&.to_i
TERRAIN_COLLECTION = "Terrain"

TERRAIN_CUES = %w[
  terrain scenery building buildings ruin ruins fortification wall walls barricade
  barricades tower towers bunker bunkers gate door doors bridge walkway platform
  emplacement turret objective monument container containers pipe pipes conduit
  generator statue statues manufactorum sanctum sector scatter fence crate barrel
  sandbag sandbags rubble trench
].to_set

# How a model's tags (our datapackage "keywords") are read. tag_list is the
# acts-as-taggable-on accessor; swap to `model.tags.map(&:name)` (or the real
# context) if the source differs.
def keywords_for(model)
  return Array(model.tag_list) if model.respond_to?(:tag_list)
  model.tags.map(&:name)
end

# The collection a model should belong to: its "faction:" tag, else "Terrain"
# when it looks like terrain, else nil (leave Unassigned).
def wanted_collection(model)
  kws = keywords_for(model)
  faction = kws.find { |k| k.to_s.downcase.start_with?("faction:") } # "faction_theme:" won't match
  return faction.split(":", 2).last.strip if faction

  mt = kws.find { |k| k.to_s.downcase.start_with?("model_type:") }
  is_terrain = (mt && mt.split(":", 2).last.strip.casecmp?("terrain")) ||
               kws.any? { |k| k.to_s.downcase.split(/[^a-z0-9]+/).any? { |t| TERRAIN_CUES.include?(t) } }
  is_terrain ? TERRAIN_COLLECTION : nil
end

# Case-insensitive find-or-create so we never make a duplicate collection.
def collection_named(name, cache)
  key = name.downcase
  cache[key] ||= Collection.where("lower(name) = ?", key).first || Collection.create!(name: name)
end

puts "=== #{DRY_RUN ? 'DRY RUN — no writes' : 'APPLYING'}#{LIMIT ? " (first #{LIMIT} models)" : ''} ==="

stats    = Hash.new(0)
examples = []
cache    = {}
seen     = 0

Model.includes(:collections).find_each do |model|
  break if LIMIT && seen >= LIMIT
  seen += 1
  begin
    if model.collections.any?
      stats[:already] += 1
      next
    end
    val = wanted_collection(model)
    if val.nil? || val.strip.empty?
      stats[:no_tag] += 1
      next
    end
    stats[:terrain_fallback] += 1 if val.casecmp?(TERRAIN_COLLECTION)
    unless DRY_RUN
      coll = collection_named(val, cache)
      model.collections << coll unless model.collections.include?(coll)
    end
    stats[:assigned] += 1
    examples << [model.name || model.id, val] if examples.size < 15
  rescue => e
    stats[:errors] += 1
    warn "  error on model #{model.id} (#{model.name rescue '?'}): #{e.class}: #{e.message}"
  end
end

puts
puts "models examined         : #{seen}"
puts "#{DRY_RUN ? 'would assign' : 'assigned'}            : #{stats[:assigned]}"
puts "  of which terrain->Terrain: #{stats[:terrain_fallback]}"
puts "already in a collection : #{stats[:already]}"
puts "no faction/terrain tag  : #{stats[:no_tag]}"
puts "errors                  : #{stats[:errors]}"
if DRY_RUN
  puts "\nsample (model -> collection):"
  examples.each { |name, coll| puts "  #{name}  ->  #{coll}" }
  puts "\nLooks right? Re-run with DRY_RUN=false to apply."
end
