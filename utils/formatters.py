"""Funções de formatação de dados"""
from typing import Union

def formatar_moeda(valor: Union[int, float], decimais: int = 2) -> str:
    """
    Formata valor como moeda USD.
    
    Args:
        valor: Valor numérico
        decimais: Número de casas decimais
        
    Returns:
        String formatada como moeda
    """
    return f"${valor:,.{decimais}f}"

def formatar_percentual(valor: float, decimais: int = 1) -> str:
    """
    Formata valor como percentual.
    
    Args:
        valor: Valor numérico (0.5 = 50%)
        decimais: Número de casas decimais
        
    Returns:
        String formatada como percentual
    """
    return f"{valor * 100:.{decimais}f}%"

def formatar_numero(valor: Union[int, float]) -> str:
    """
    Formata número com separadores de milhares.
    
    Args:
        valor: Valor numérico
        
    Returns:
        String formatada
    """
    return f"{valor:,}"
