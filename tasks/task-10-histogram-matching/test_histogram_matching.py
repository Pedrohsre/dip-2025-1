#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import os

# Importar a função diretamente do arquivo
exec(open('task-07-histogram-matching.py').read())

def test_histogram_matching():
    """
    Teste simples que carrega as imagens, aplica histogram matching e salva o resultado
    """
    # Obter o diretório atual do script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Caminhos das imagens
    source_path = os.path.join(current_dir, 'source.jpg')
    reference_path = os.path.join(current_dir, 'reference.jpg')
    output_path = os.path.join(current_dir, 'resultado.jpg')
    
    # Verificar se as imagens existem
    if not os.path.exists(source_path):
        print(f"Erro: Arquivo {source_path} não encontrado!")
        return False
    
    if not os.path.exists(reference_path):
        print(f"Erro: Arquivo {reference_path} não encontrado!")
        return False
    
    try:
        # Carregar as imagens
        print("Carregando imagens...")
        source_img = cv.imread(source_path)
        reference_img = cv.imread(reference_path)
        
        if source_img is None:
            print(f"Erro: Não foi possível carregar {source_path}")
            return False
        
        if reference_img is None:
            print(f"Erro: Não foi possível carregar {reference_path}")
            return False
        
        # Converter de BGR para RGB (OpenCV carrega em BGR por padrão)
        source_img_rgb = cv.cvtColor(source_img, cv.COLOR_BGR2RGB)
        reference_img_rgb = cv.cvtColor(reference_img, cv.COLOR_BGR2RGB)
        
        print(f"Imagem source: {source_img_rgb.shape}")
        print(f"Imagem reference: {reference_img_rgb.shape}")
        
        # Aplicar histogram matching
        print("Aplicando histogram matching...")
        matched_img = match_histograms_rgb(source_img_rgb, reference_img_rgb)
        
        # Converter de volta para BGR para salvar com OpenCV
        matched_img_bgr = cv.cvtColor(matched_img, cv.COLOR_RGB2BGR)
        
        # Salvar o resultado
        success = cv.imwrite(output_path, matched_img_bgr)
        
        if success:
            print(f"Resultado salvo em: {output_path}")
            return True
        else:
            print("Erro ao salvar o resultado!")
            return False
        
    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        return False

if __name__ == "__main__":
    print("=== Teste de Histogram Matching ===")
    success = test_histogram_matching()
    
    if success:
        print("✅ Teste executado com sucesso!")
    else:
        print("❌ Teste falhou!")
