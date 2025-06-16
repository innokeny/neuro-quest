from src.engine.engine import Engine, EngineConfig
from src.ml.inference.master import MasterConfig, GenerationConfig
from pathlib import Path

config = EngineConfig(
    vector_db_path=Path('tmp/db'),
    number_of_remind_items=5,
    master_config=MasterConfig(
        path=Path('Qwen/Qwen3-0.6B').as_posix(),
        preambular='''
**The Tale of the Stolen Star**  

### **Player Character**  
- **Race:**Human  
- **Class:** Rogue (Thief)  
- **Inventory:** Lockpicks, a silver dagger (engraved with a crescent moon), a hooded cloak lined with hidden pockets, a stolen signet ring (unbeknownst to him, cursed), and a small pouch of glowing blue sand.  
- **Abilities:** Expert lockpicker, nimble fingers, stealth mastery, and an uncanny ability to sense traps.  
- **Backstory:** Born in the slums of **Hollow’s End**, **Riley Quickfingers** learned early that survival meant taking what others wouldn’t miss—or wouldn’t notice. Though he steals, he has a code: never from those who can’t afford to lose it. His latest job, however, might have crossed a line he didn’t see coming.  

---

### **Key NPCs**  

1. **Booster "The Barrel" Durnan**  
   - **Role:** Tavernkeeper of **Booster’s Tavern**, retired thief.  
   - **Appearance:** A burly human with a shaved head, a thick black beard, and a permanent smirk. His arms are covered in faded jailhouse tattoos.  
   - **Personality:** Gruff but fair, with a soft spot for fellow rogues who show potential. Hates cheats and bullies.  
   - **Motive:** Runs a neutral ground for thieves and informants but keeps a tight leash on trouble. Knows more than he lets on.  

2. **Seraphine Duskwhisper**  
   - **Role:** A mysterious elven scholar searching for lost artifacts.  
   - **Appearance:** Tall, pale, with silver hair tied in intricate braids. Wears a long, dark-blue coat lined with arcane symbols.  
   - **Personality:** Coldly polite, always calculating. Speaks in riddles when annoyed.  
   - **Motive:** Hired Riley to retrieve an artifact—**The Star of Luminis**—but didn’t mention its dangerous nature.  

3. **Gristle the Snitch**  
   - **Role:** A weaselly street informant.  
   - **Appearance:** A scrawny half-elf with a lazy eye and a habit of chewing his nails.  
   - **Personality:** Nervous, twitchy, and always looking over his shoulder.  
   - **Motive:** Will sell anyone out for the right price but knows every secret in town.  

---

### **Enemies**  

1. **The Black Hand Syndicate**  
   - **Type:** Organized thieves' guild.  
   - **Appearance:** Black leather armor, red scarves, and silver daggers.  
   - **Combat Behavior:** Fight dirty—ambushes, poison, and overwhelming numbers.  
   - **Objective:** Retrieve the stolen **Star of Luminis** and silence Riley.  

2. **The Hollow Specter**  
   - **Type:** A shadowy entity bound to the cursed ring Riley carries.  
   - **Appearance:** A shifting, humanoid figure made of smoke and whispers.  
   - **Combat Behavior:** Phases through walls, drains strength with a touch.  
   - **Objective:** Reclaim the ring—or claim Riley’s soul instead.  

---

### **Main Location: Hollow’s End**  
A crumbling port city built atop ancient ruins, where the wealthy live in marble towers while the poor scurry through sewers and alleyways. The city thrives on secrets, and the **Star of Luminis**—a relic said to reveal hidden truths—has just been stolen from the Syndicate’s vault. Now, the underworld is in chaos.  

---

### **Sub-Locations**  

1. **Booster’s Tavern**  
   - A dimly lit den of ale and intrigue. The air smells of roasted meat and spilled mead. The walls are covered in wanted posters, and the floorboards creak with hidden compartments beneath.  

2. **The Shivering Market**  
   - A black-market bazaar where stolen goods change hands. The scent of exotic spices mixes with the metallic tang of smuggled weapons. Shadows move unnaturally here.  

3. **The Syndicate’s Den**  
   - A fortified gambling hall with red velvet curtains and rigged games. Guards lurk in every corner, watching for intruders.  

4. **The Whispering Catacombs**  
   - Beneath the city, these tunnels are lined with skulls that seem to murmur secrets to those who listen.  

5. **The Clocktower of Old Veyne**  
   - A rusted, ancient tower where Seraphine conducts her research. The gears inside hum with latent magic.  

---

### **Artifacts**  

1. **The Star of Luminis**  
   - A palm-sized crystal shard that glows faintly blue. Reveals hidden messages when held under moonlight.  

2. **The Cursed Signet Ring**  
   - Silver, with a black onyx stone. Whispers names of the dead at midnight.  

3. **The Gauntlet of Shadows**  
   - A blackened steel glove that lets the wearer phase through objects briefly.  

4. **The Tome of Whispers**  
   - A book that writes itself with secrets overheard nearby.  

5. **The Lantern of the Lost**  
   - When lit, reveals invisible creatures—but also attracts them.  

6. **The Dagger of Echoes**  
   - Strikes silently but leaves behind phantom sounds of past killings.  

7. **The Mask of a Thousand Faces**  
   - Changes the wearer’s appearance—but sometimes, the faces linger too long.  

---

### **The Story Unfolds…**  
Riley thought stealing the **Star of Luminis** would be just another job. But now, the Syndicate wants it back, the Hollow Specter stalks him, and Seraphine’s true motives are unclear. The only safe place left is **Booster’s Tavern**—but even there, trust is a currency spent quickly.  

Will Riley uncover the Star’s secret before the city’s shadows swallow him whole?
''',
        generation_config=GenerationConfig(temperature=0.7, max_new_tokens=128),
    ),
    ner_model_path=Path('models/ner'),
    embedding_model_path=Path(
        'sentence-transformers/all-MiniLM-L6-v2'
    )
)

engine = Engine(config, debug=True) 