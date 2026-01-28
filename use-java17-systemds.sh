#!/bin/bash
# ------------------------------------------------------------------
# SystemDS macOS Build-Skript
# Setzt JAVA_HOME, PATH, Maven und erzeugt systemds-standalone.sh
# ------------------------------------------------------------------

# 1️⃣ Setze Java 17
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH="$JAVA_HOME/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"

# 2️⃣ Optional: Python, ghcup, uix/Deno, Coursier, JetBrains Toolbox
export PATH="/Library/Frameworks/Python.framework/Versions/3.11/bin:$HOME/.ghcup/bin:$HOME/.uix/bin:$PATH"
export DENO_INSTALL="$HOME/.uix"
export PATH="$DENO_INSTALL/bin:$PATH"
export PATH="$PATH:/Users/mori/Library/Application Support/Coursier/bin"
export PATH="$PATH:/Users/mori/Library/Application Support/JetBrains/Toolbox/scripts"

# 3️⃣ Prüfen, ob Maven existiert
if ! command -v mvn >/dev/null 2>&1; then
  echo "ERROR: Maven (mvn) nicht gefunden. Bitte installieren!"
  exit 1
fi

# 4️⃣ Prüfen, ob wir im Projekt-Root sind (pom.xml vorhanden)
if [ ! -f "pom.xml" ]; then
  echo "ERROR: pom.xml nicht gefunden. Bitte ins SystemDS-Projekt-Root wechseln."
  exit 1
fi

# 5️⃣ Maven Build ausführen
echo "📦 Starte Maven Build..."
mvn clean package -DskipTests

# 6️⃣ Standalone-Skript erzeugen
echo "🔧 Erzeuge bin/systemds-standalone.sh..."

mkdir -p bin
cat > bin/systemds-standalone.sh << 'EOF'
#!/bin/bash
# Standalone-Launcher für SystemDS

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
JAR_FILE="$SCRIPT_DIR/../target/systemds-3.4.0-SNAPSHOT.jar"

if [ ! -f "$JAR_FILE" ]; then
  echo "ERROR: Standalone JAR nicht gefunden: $JAR_FILE"
  exit 1
fi

java -cp "$JAR_FILE" org.apache.sysds.api.DMLScript "$@"
EOF

# 7️⃣ Ausführbar machen
chmod +x bin/systemds-standalone.sh

echo "✅ Fertig! Standalone-Skript erstellt: bin/systemds-standalone.sh"

