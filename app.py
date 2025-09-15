import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyteomics import mzml
import io

def extract_precursor_data(mzml_file):
    """
    Extract precursor m/z values and charge states from mzML file.
    
    Returns:
        tuple: (precursor_mz_list, charge_state_list, neutral_mass_list)
    """
    precursor_mz = []
    charge_states = []
    neutral_masses = []
    
    with mzml.read(mzml_file) as reader:
        for spectrum in reader:
            # Check if this is an MS2 spectrum with precursor info
            if spectrum.get('ms level') == 2:
                precursor_list = spectrum.get('precursorList', {})
                precursors = precursor_list.get('precursor', [])
                
                for precursor in precursors:
                    selected_ions = precursor.get('selectedIonList', {}).get('selectedIon', [])
                    
                    for ion in selected_ions:
                        # Extract m/z
                        mz = ion.get('selected ion m/z')
                        # Extract charge state
                        charge = ion.get('charge state')
                        
                        if mz is not None and charge is not None:
                            precursor_mz.append(mz)
                            charge_states.append(charge)
                            # Calculate neutral mass: (m/z * charge) - (charge * proton_mass)
                            # Proton mass = 1.007276466812 Da
                            neutral_mass = (mz * charge) - (charge * 1.007276466812)
                            neutral_masses.append(neutral_mass)
    
    return precursor_mz, charge_states, neutral_masses

def create_histogram_plots(precursor_mz, charge_states, neutral_masses, filename, bins=50):
    """
    Create three histogram plots for m/z, neutral mass, and charge states.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # m/z histogram
    axes[0].hist(precursor_mz, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Precursor m/z')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Precursor m/z Distribution\n({filename})')
    axes[0].grid(True, alpha=0.3)
    
    # Neutral mass histogram
    axes[1].hist(neutral_masses, bins=bins, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('Neutral Mass (Da)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Neutral Mass Distribution\n({filename})')
    axes[1].grid(True, alpha=0.3)
    
    # Charge state histogram
    unique_charges = sorted(set(charge_states))
    charge_bins = np.arange(min(unique_charges) - 0.5, max(unique_charges) + 1.5, 1)
    axes[2].hist(charge_states, bins=charge_bins, alpha=0.7, color='salmon', edgecolor='black')
    axes[2].set_xlabel('Charge State')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'Charge State Distribution\n({filename})')
    axes[2].set_xticks(unique_charges)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def get_summary_stats(precursor_mz, charge_states, neutral_masses):
    """
    Calculate summary statistics for the data.
    """
    stats = {
        'Total MS2 Spectra': len(precursor_mz),
        'Precursor m/z Range': f"{min(precursor_mz):.2f} - {max(precursor_mz):.2f}",
        'Neutral Mass Range': f"{min(neutral_masses):.2f} - {max(neutral_masses):.2f} Da",
        'Charge States Found': sorted(set(charge_states)),
        'Mean m/z': f"{np.mean(precursor_mz):.2f}",
        'Mean Neutral Mass': f"{np.mean(neutral_masses):.2f} Da"
    }
    return stats

# Streamlit App
def main():
    st.set_page_config(page_title="mzML Precursor Analysis", layout="wide")
    
    st.title("mzML Precursor Mass and Charge State Analysis")
    st.write("Upload one or more mzML files to analyze precursor masses and charge states")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose mzML files", 
        type=['mzml', 'mzML'], 
        accept_multiple_files=True,
        help="Upload one or more mzML files for analysis"
    )
    
    if uploaded_files:
        # Sidebar for plot customization
        st.sidebar.header("Plot Settings")
        bins = st.sidebar.slider("Number of bins for histograms", 10, 100, 50)
        
        for i, uploaded_file in enumerate(uploaded_files):
            st.header(f"Analysis for: {uploaded_file.name}")
            
            try:
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Reading mzML file...")
                progress_bar.progress(25)
                
                # Extract data
                precursor_mz, charge_states, neutral_masses = extract_precursor_data(uploaded_file)
                progress_bar.progress(50)
                
                if not precursor_mz:
                    st.warning(f"No MS2 precursor data found in {uploaded_file.name}")
                    continue
                
                status_text.text("Creating plots...")
                progress_bar.progress(75)
                
                # Create plots
                fig = create_histogram_plots(
                    precursor_mz, charge_states, neutral_masses, 
                    uploaded_file.name, bins
                )
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                
                # Display plots
                st.pyplot(fig)
                
                # Create two columns for summary stats and data table
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display summary statistics
                    st.subheader("Summary Statistics")
                    stats = get_summary_stats(precursor_mz, charge_states, neutral_masses)
                    for key, value in stats.items():
                        st.metric(key, value)
                
                with col2:
                    # Create downloadable data
                    st.subheader("Download Data")
                    df = pd.DataFrame({
                        'Precursor_mz': precursor_mz,
                        'Charge_State': charge_states,
                        'Neutral_Mass_Da': neutral_masses
                    })
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=f"Download data as CSV ({uploaded_file.name})",
                        data=csv,
                        file_name=f"{uploaded_file.name}_precursor_data.csv",
                        mime="text/csv"
                    )
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Add separator between files
                if i < len(uploaded_files) - 1:
                    st.divider()
                    
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                st.write("Please ensure the file is a valid mzML format.")
    
    else:
        st.info("Please upload one or more mzML files to begin analysis.")
        
        # Show example of what the app will do
        st.subheader("What this app does:")
        st.write("""
        1. **Upload mzML files**: Select one or more mzML files from your computer
        2. **Extract precursor data**: Reads MS2 spectra and extracts:
           - Precursor m/z values
           - Charge states  
           - Calculated neutral masses
        3. **Generate plots**: Creates three histograms for each file:
           - Precursor m/z distribution
           - Neutral mass distribution  
           - Charge state distribution
        4. **Summary statistics**: Shows key metrics about your data
        5. **Download data**: Export extracted data as CSV files
        """)

if __name__ == "__main__":
    main()