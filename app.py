import streamlit as st
import pandas as pd
import numpy as np
import requests
import urllib.parse
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.ML.Descriptors import MoleculeDescriptors

# ==========================================
# CONFIGURACIÓN DE INTERFAZ Y ESTILOS (UI/UX)
# ==========================================
st.set_page_config(
    page_title="QSPR ESOL Explorer | RDKit",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .iupac-box {
        background-color: #f1f5f9;
        border-left: 5px solid #0284c7;
        padding: 12px 16px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.95rem;
        color: #0f172a;
        margin-bottom: 12px;
        word-wrap: break-word;
    }
    .smiles-box {
        background-color: #fdf2f8;
        border-left: 5px solid #db2777;
        padding: 12px 16px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.95rem;
        color: #831843;
        margin-bottom: 12px;
        word-wrap: break-word;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# FUNCIONES DE DATOS Y QUIMIOINFORMÁTICA
# ==========================================
URL_ESOL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"

DESCRIPTORES_CURSO = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", 
    "NumRotatableBonds", "HeavyAtomCount", "RingCount", "FractionCSP3"
]

@st.cache_data(show_spinner="Descargando dataset ESOL/Delaney desde DeepChem...")
def cargar_dataset_esol():
    df = pd.read_csv(URL_ESOL)
    return df

@st.cache_data(show_spinner=False)
def obtener_iupac_pubchem(smiles: str) -> str:
    """Consulta el nombre IUPAC oficial desde la API de PubChem a partir del SMILES."""
    try:
        smiles_encoded = urllib.parse.quote(smiles)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles_encoded}/property/IUPACName/JSON"
        response = requests.get(url, timeout=4)
        if response.status_code == 200:
            data = response.json()
            return data["PropertyTable"]["Properties"][0].get("IUPACName", "No registrado en PubChem")
        return "Nombre IUPAC no disponible en PubChem para este confórmero/estructura"
    except Exception:
        return "Sin conexión temporal a PubChem API"

@st.cache_data(show_spinner=False)
def calcular_todos_descriptores(smiles: str):
    """Calcula la lista completa de los ~217 descriptores moleculares de RDKit para 1 SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    nombres = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(nombres)
    valores = calc.CalcDescriptors(mol)
    df_res = pd.DataFrame({
        "ID": range(1, len(nombres) + 1),
        "Descriptor_RDKit": nombres,
        "Valor": valores
    })
    df_res["En_Sesion_2"] = df_res["Descriptor_RDKit"].apply(
        lambda x: "⭐ Sí (Curso)" if x in DESCRIPTORES_CURSO else "Completo RDKit"
    )
    return df_res

@st.cache_data(show_spinner="Calculando los descriptores de RDKit para el lote seleccionado...")
def calcular_lote_descriptores(smiles_series):
    nombres = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(nombres)
    matriz = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            matriz.append(calc.CalcDescriptors(mol))
        else:
            matriz.append([np.nan] * len(nombres))
    return pd.DataFrame(matriz, columns=nombres)

# ==========================================
# CARGA INICIAL DE DATOS
# ==========================================
df_esol = cargar_dataset_esol()
col_sol_exp = "measured log solubility in mols per litre"
col_sol_pred = "ESOL predicted log solubility in mols per litre"

# ==========================================
# BARRA LATERAL (FILTROS E INTERACTIVIDAD)
# ==========================================
with st.sidebar:
    st.header("🎛️ Controles de Exploración")
    st.caption("Sesión 2: QSPR y Descriptores Moleculares (Puntos 1 al 9)")
    
    # Buscador por texto
    busqueda_texto = st.text_input("🔎 Buscar por Nombre o SMILES:", placeholder="Ej. Caffeine, Ethanol, c1ccccc1")
    
    # Sliders de filtrado químico sobre el dataset ESOL
    min_mw, max_mw = float(df_esol["Molecular Weight"].min()), float(df_esol["Molecular Weight"].max())
    rango_mw = st.slider("Masa Molecular (g/mol):", min_mw, max_mw, (min_mw, max_mw))
    
    min_sol, max_sol = float(df_esol[col_sol_exp].min()), float(df_esol[col_sol_exp].max())
    rango_sol = st.slider("Solubilidad Exp. (log S):", min_sol, max_sol, (min_sol, max_sol))

    # Aplicar filtros al DataFrame
    df_filtrado = df_esol[
        (df_esol["Molecular Weight"] >= rango_mw[0]) &
        (df_esol["Molecular Weight"] <= rango_mw[1]) &
        (df_esol[col_sol_exp] >= rango_sol[0]) &
        (df_esol[col_sol_exp] <= rango_sol[1])
    ].copy()
    
    if busqueda_texto:
        df_filtrado = df_filtrado[
            df_filtrado["Compound ID"].str.contains(busqueda_texto, case=False, na=False) |
            df_filtrado["smiles"].str.contains(busqueda_texto, case=False, na=False)
        ]

    st.info(f"🧪 **{len(df_filtrado)}** de **{len(df_esol)}** moléculas filtradas.")
    
    st.divider()
    st.subheader("🖼️ Opciones de Renderizado 2D")
    res_img = st.slider("Tamaño de estructura (px):", 250, 600, 380, step=30)
    mostrar_idx_atomos = st.toggle("Mostrar numeración de átomos", value=False)
    mostrar_estereo = st.toggle("Resaltar estereoquímica", value=True)
    precision_dec = st.slider("Decimales en tablas:", 2, 6, 4)

# ==========================================
# ENCABEZADO PRINCIPAL
# ==========================================
st.title("🧪 Explorador QSPR de Solubilidad Acuosa (Dataset ESOL/Delaney)")
st.markdown("""
**Flujo de trabajo hasta el Punto 9:** `Estructura Molecular` ➔ `SMILES` ➔ `RDKit` ➔ `217 Descriptores Moleculares`  
Explora las moléculas del dataset de Delaney, consulta su nomenclatura **IUPAC**, inspecciona su topología y extrae la matriz completa de descriptores para tus análisis.
""")

# Pestañas de navegación UI
tab_inspector, tab_espacio, tab_batch, tab_teoria = st.tabs([
    "🔬 1. Inspector Molecular y 217 Descriptores",
    "🗺️ 2. Espacio Químico Interactivo (ESOL)",
    "⚙️ 3. Generador de Matriz Completa (Punto 9)",
    "📖 4. Fundamentos Teóricos (Puntos 1–4)"
])

# ==========================================
# PESTAÑA 1: INSPECTOR MOLECULAR Y 217 DESCRIPTORES
# ==========================================
with tab_inspector:
    if df_filtrado.empty:
        st.warning("No se encontraron moléculas con los filtros actuales. Ajusta los deslizadores en la barra lateral.")
    else:
        col_sel, col_nav = st.columns([3, 1])
        
        with col_sel:
            opciones_mol = [
                f"[{idx}] {row['Compound ID']}  |  SMILES: {row['smiles']}  |  logS: {row[col_sol_exp]}"
                for idx, row in df_filtrado.iterrows()
            ]
            mol_elegida = st.selectbox("Selecciona un compuesto del dataset ESOL:", opciones_mol)
            idx_actual = int(mol_elegida.split("]")[0].replace("[", ""))
            fila = df_esol.loc[idx_actual]
            
        with col_nav:
            st.write("")
            st.write("")
            usar_canonical = st.checkbox("Convertir a Canonical SMILES", value=True)

        smiles_raw = str(fila["smiles"])
        mol_rdkit = Chem.MolFromSmiles(smiles_raw)
        smiles_mostrar = Chem.MolToSmiles(mol_rdkit, canonical=True) if (mol_rdkit and usar_canonical) else smiles_raw

        # Consulta de Nombre IUPAC en PubChem
        iupac_nombre = obtener_iupac_pubchem(smiles_mostrar)

        st.divider()

        # Layout dividido: Estructura 2D a la izquierda | Identificadores y KPIs a la derecha
        c_img, c_info = st.columns([1.1, 1.9], gap="large")

        with c_img:
            st.subheader(f"📌 {fila['Compound ID']}")
            if mol_rdkit:
                mol_dibujo = Chem.Mol(mol_rdkit)
                if mostrar_idx_atomos:
                    for atom in mol_dibujo.GetAtoms():
                        atom.SetAtomMapNum(atom.GetIdx())
                
                opts = Draw.MolDrawOptions()
                opts.addStereoAnnotation = mostrar_estereo
                img_mol = Draw.MolToImage(mol_dibujo, size=(res_img, res_img), options=opts)
                st.image(img_mol, use_container_width=False)
            else:
                st.error("Estructura SMILES inválida.")

        with c_info:
            st.subheader("🏷️ Nomenclatura y Representación Lineal")
            
            st.markdown("**Nombre Común / ID en ESOL:**")
            st.markdown(f"#### `{fila['Compound ID']}`")
            
            st.markdown("**Nombre Sistemático IUPAC (vía PubChem):**")
            st.markdown(f'<div class="iupac-box">{iupac_nombre}</div>', unsafe_allow_html=True)
            
            st.markdown("**Código SMILES:**")
            st.markdown(f'<div class="smiles-box">{smiles_mostrar}</div>', unsafe_allow_html=True)

            # Tarjetas de Solubilidad Experimental vs Predicha por Delaney
            st.markdown("##### 💧 Propiedad Objetivo: Solubilidad Acuosa ($\log S$)")
            k1, k2, k3, k4 = st.columns(4)
            sol_exp = fila[col_sol_exp]
            sol_esol = fila[col_sol_pred]
            concentracion_molar = 10 ** sol_exp
            
            k1.metric("log S Experimental", f"{sol_exp:.3f} mol/L")
            k2.metric("log S Estimado (ESOL)", f"{sol_esol:.3f} mol/L", delta=f"{sol_esol - sol_exp:.3f}")
            k3.metric("Solubilidad Molar (S)", f"{concentracion_molar:.2e} M")
            k4.metric("Peso Molecular", f"{fila['Molecular Weight']:.2f} g/mol")

        # Cálculo de los 217 Descriptores de RDKit
        if mol_rdkit:
            df_217 = calcular_todos_descriptores(smiles_mostrar)
            
            st.divider()
            st.subheader("🎯 Los 9 Descriptores Interpretables de la Sesión 2")
            df_9 = df_217[df_217["Descriptor_RDKit"].isin(DESCRIPTORES_CURSO)].set_index("Descriptor_RDKit")["Valor"]
            
            d_cols = st.columns(9)
            for i, desc_name in enumerate(DESCRIPTORES_CURSO):
                val = df_9.get(desc_name, np.nan)
                formato = f"{int(val)}" if desc_name in ["NumHDonors", "NumHAcceptors", "NumRotatableBonds", "HeavyAtomCount", "RingCount"] else f"{val:.2f}"
                d_cols[i].metric(desc_name, formato)

            st.divider()
            st.subheader(f"🧬 Explorador de los {len(df_217)} Descriptores Moleculares de RDKit")
            
            f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
            with f_col1:
                buscar_desc = st.text_input("Filtrar descriptor por nombre (ej. Chi, Kappa, VSA, EState, Lipinski):", "")
            with f_col2:
                filtro_cat = st.selectbox("Categoría:", ["Todos los descriptores", "Solo los 9 del curso", "Excluir valores en 0"])
            with f_col3:
                orden_val = st.selectbox("Ordenar por:", ["ID Original RDKit", "Nombre (A-Z)", "Mayor Valor", "Menor Valor"])

            df_vista_desc = df_217.copy()
            if buscar_desc:
                df_vista_desc = df_vista_desc[df_vista_desc["Descriptor_RDKit"].str.contains(buscar_desc, case=False, na=False)]
            if filtro_cat == "Solo los 9 del curso":
                df_vista_desc = df_vista_desc[df_vista_desc["Descriptor_RDKit"].isin(DESCRIPTORES_CURSO)]
            elif filtro_cat == "Excluir valores en 0":
                df_vista_desc = df_vista_desc[df_vista_desc["Valor"] != 0]

            if orden_val == "Nombre (A-Z)":
                df_vista_desc = df_vista_desc.sort_values("Descriptor_RDKit")
            elif orden_val == "Mayor Valor":
                df_vista_desc = df_vista_desc.sort_values("Valor", ascending=False)
            elif orden_val == "Menor Valor":
                df_vista_desc = df_vista_desc.sort_values("Valor", ascending=True)

            df_vista_desc["Valor"] = df_vista_desc["Valor"].round(precision_dec)
            
            st.dataframe(
                df_vista_desc,
                use_container_width=True,
                height=380,
                hide_index=True
            )
            
            # Descarga individual
            csv_ind = df_vista_desc.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 Descargar los {len(df_217)} descriptores de {fila['Compound ID']} (.CSV)",
                data=csv_ind,
                file_name=f"descriptores_{fila['Compound ID'].replace(' ', '_')}.csv",
                mime="text/csv"
            )

# ==========================================
# PESTAÑA 2: ESPACIO QUÍMICO INTERACTIVO
# ==========================================
with tab_espacio:
    st.subheader("🗺️ Mapa del Espacio Químico y Dominio de Aplicabilidad (ESOL)")
    st.markdown("Explora cómo se distribuyen las moléculas filtradas según sus descriptores y su solubilidad experimental ($\log S$).")
    
    g_col1, g_col2, g_col3 = st.columns(3)
    vars_numericas = [
        "Molecular Weight", "Polar Surface Area", "Number of Rotatable Bonds",
        "Number of H-Bond Donors", "Number of Rings", "Minimum Degree",
        col_sol_exp, col_sol_pred
    ]
    with g_col1:
        eje_x = st.selectbox("Eje X:", vars_numericas, index=0)
    with g_col2:
        eje_y = st.selectbox("Eje Y:", vars_numericas, index=6)
    with g_col3:
        color_var = st.selectbox("Colorear por:", vars_numericas, index=1)

    fig = px.scatter(
        df_filtrado,
        x=eje_x,
        y=eje_y,
        color=color_var,
        hover_name="Compound ID",
        hover_data=["smiles", "Molecular Weight", "Polar Surface Area", col_sol_exp],
        color_continuous_scale="Viridis",
        height=520,
        title=f"Relación entre {eje_x} y {eje_y}"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PESTAÑA 3: CÁLCULO MASIVO (HASTA PUNTO 9)
# ==========================================
with tab_batch:
    st.subheader("⚙️ Construcción del Dataset con los 217 Descriptores de RDKit")
    st.markdown("""
    En los puntos 7, 8 y 9 del cuaderno se construye la tabla de descriptores a partir de la columna `smiles` para su posterior uso como matriz predictora $X$. Aquí puedes generar la matriz con los **9 descriptores del curso** o expandirla a los **217 descriptores de RDKit**.
    """)
    
    c_b1, c_b2 = st.columns(2)
    with c_b1:
        alcance_lote = st.radio(
            "Conjunto de moléculas a procesar:",
            [f"Solo las primeras 100 moléculas (Rápido)", f"Moléculas filtradas actualmente ({len(df_filtrado)})", f"Todo el dataset ESOL ({len(df_esol)} moléculas)"]
        )
    with c_b2:
        tipo_matriz = st.radio(
            "Cantidad de descriptores por molécula:",
            ["Los 217 descriptores completos de RDKit", "Solo los 9 descriptores de la Sesión 2"]
        )

    if st.button("🚀 Generar Matriz de Descriptores RDKit", type="primary"):
        if "100" in alcance_lote:
            df_target = df_esol.head(100).copy()
        elif "filtradas" in alcance_lote:
            df_target = df_filtrado.copy()
        else:
            df_target = df_esol.copy()

        df_desc_lote = calcular_lote_descriptores(df_target["smiles"])
        
        if tipo_matriz == "Solo los 9 descriptores de la Sesión 2":
            df_desc_lote = df_desc_lote[DESCRIPTORES_CURSO]

        df_final_qspr = pd.concat([
            df_target[["Compound ID", "smiles", col_sol_exp]].reset_index(drop=True),
            df_desc_lote.round(precision_dec).reset_index(drop=True)
        ], axis=1)

        st.success(f"✅ Matriz QSPR construida exitosamente: **{df_final_qspr.shape[0]} filas × {df_final_qspr.shape[1]} columnas**.")
        st.dataframe(df_final_qspr, use_container_width=True, height=420)

        csv_batch = df_final_qspr.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar Matriz QSPR Completa (.CSV)",
            data=csv_batch,
            file_name="ESOL_RDKit_217_Descriptores.csv",
            mime="text/csv"
        )

# ==========================================
# PESTAÑA 4: RESUMEN DIDÁCTICO (PUNTOS 1 AL 4)
# ==========================================
with tab_teoria:
    st.subheader("📖 Síntesis de Conceptos de la Sesión 2")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("""
        #### 1. Enfoque QSPR y Solubilidad Acuosa
        * **Solubilidad ($\log S$):** Logaritmo base 10 de la solubilidad molar en agua ($\text{mol/L}$).
        * **QSPR (*Quantitative Structure–Property Relationship*):** Relaciona cuantitativamente la estructura molecular con una propiedad fisicoquímica antes de realizar ensayos de laboratorio.
        
        #### 2. Notación SMILES
        * **Átomos alifáticos vs. aromáticos:** Mayúsculas (`C`, `O`, `N`) frente a minúsculas (`c1ccccc1`).
        * **Enlaces y ramificaciones:** Simple (`-` u omitido), doble (`=`), triple (`#`), ramificaciones con paréntesis `()`.
        """)
    with col_t2:
        st.markdown("""
        #### 3. Los 9 Descriptores Base del Caso
        * `MolWt`: Masa molecular promedio.
        * `MolLogP`: Coeficiente de partición octanol/agua (hidrofobicidad de Crippen).
        * `TPSA`: Área superficial polar topológica.
        * `NumHDonors` / `NumHAcceptors`: Donadores y aceptores de puentes de hidrógeno.
        * `NumRotatableBonds`: Flexibilidad conformacional.
        * `HeavyAtomCount`, `RingCount`, `FractionCSP3`: Tamaño pesado, ciclos y saturación $sp^3$.
        """)
