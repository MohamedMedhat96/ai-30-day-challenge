from anthropic import Anthropic

client = Anthropic()

with client.messages.stream( # With is used to handle resources to open and close them (files, stream, io)
    model="claude-haiku-4-5", max_tokens=1500, temperature=0.5,
    system="""<rules>
  <rule>Prioritise quality over obscurity — only recommend an album genuinely worth listening to.</rule>
  <rule>Avoid albums with recent mainstream attention, TikTok moments, Netflix placements, or critical reappraisals in the last 1-2 years.</rule>
  <rule>Never repeat an album from the never_recommend list.</rule>
  <rule>Return ONLY a valid JSON object with no preamble and no markdown fences.</rule>
  <rule>Do not use an em dash anywhere in why_this_album.</rule>
</rules>

<output_schema>
  <field name="album_title" type="string"/>
  <field name="artist" type="string"/>
  <field name="year" type="integer"/>
  <field name="genre" type="string"/>
  <field name="album_description" type="string" note="2-3 sentences, objective"/>
  <field name="why_this_album" type="string" note="2-3 sentences, personal to this user's taste, make it feel human not AI, no em dash"/>
  <field name="highlight_track" type="string"/>
  <field name="highlight_track_reason" type="string" note="one sentence"/>
  <field name="spotify_search_query" type="string"/>
</output_schema>

<example>
  <user_profile>
    <loved_albums>
      <album>In Rainbows — Radiohead</album>
    </loved_albums>
    <genre_affinities>art rock, post-rock</genre_affinities>
    <mood_tags>introspective, atmospheric</mood_tags>
    <era_lean>2000s onward</era_lean>
    <discovery_appetite>open to lesser-known artists in familiar genres</discovery_appetite>
    <never_recommend>
      <album>OK Computer — Radiohead</album>
    </never_recommend>
    <avoid_trending>
      <album>Currents — Tame Impala</album>
    </avoid_trending>
  </user_profile>
  <response>{"album_title":"Yellow House","artist":"Grizzly Bear","year":2006,"genre":"art rock","album_description":"A hushed, harmony-heavy record built from layered vocals and warm analog textures recorded in a Cape Cod house.","why_this_album":"If you love the patient builds and textured atmosphere of In Rainbows, this gives you that same slow-burn intimacy with a folkier, more intimate core.","highlight_track":"Knife","highlight_track_reason":"It captures the album's blend of close harmony and gentle tension in under four minutes."}</response>
</example>""",
    messages=[{"role": "user", "content": """<user_profile>
  <loved_albums>
    <album>Wildflower — The Avalanches</album>
    <album>Blonde — Frank Ocean</album>
    <album>Geogaddi — Boards of Canada</album>
    <album>Histoire de Melody Nelson — Serge Gainsbourg</album>
    <album>To Pimp a Butterfly — Kendrick Lamar</album>
    <album>Either/Or — Elliott Smith</album>
    <album>Source Tags & Codes — ...And You Will Know Us by the Trail of Dead</album>
    <album>Funeral — Arcade Fire</album>
    <album>Madvillainy — Madvillain</album>
    <album>The Glow Pt. 2 — The Microphones</album>
    <album>Brat - Charlie Xcxx</album>           
  </loved_albums>
  <genre_affinities>indie folk, psychedelic rock, trip hop, hip hop, art pop, ambient</genre_affinities>
  <mood_tags>nostalgic, hazy, melancholic, warm</mood_tags>
  <era_lean>90s and 2000s</era_lean>
  <discovery_appetite>open to lesser-known artists in familiar genres</discovery_appetite>
  <never_recommend>
    <album>OK Computer — Radiohead</album>
    <album>In Rainbows — Radiohead</album>
    <album>Illinois — Sufjan Stevens</album>
  </never_recommend>
  <avoid_trending>
    <album>Currents — Tame Impala</album>
    <album>Norman Fucking Rockwell! — Lana Del Rey</album>
    <album>Igor — Tyler, the Creator</album>
  </avoid_trending>
</user_profile>"""}],
) as stream:
    for text in stream.text_stream: # prints each stream
        print(text, end="", flush=True)   # flushing stream to print everything without cashing
    final = stream.get_final_message() # save final message

print("\n\nUSAGE:", final.usage) # print total usage


IN_RATE, OUT_RATE = 1.00, 5.00 
cost = final.usage.input_tokens/1e6*IN_RATE + final.usage.output_tokens/1e6*OUT_RATE #calculate cost per API
print(f"Approx cost: ${cost:.6f}")

