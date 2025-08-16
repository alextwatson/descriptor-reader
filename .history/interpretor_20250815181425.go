package main

import (
	"fmt"
	"strings"
)

func interpretDescriptor(desc string) string {
	// Remove checksum if present
	parts := strings.Split(desc, "#")
	core := parts[0]

	if strings.HasPrefix(core, "wpkh(") {
		return "Native SegWit (bech32) single key"
	}
	if strings.HasPrefix(core, "sh(wpkh(") {
		return "P2SH-wrapped SegWit single key"
	}
	if strings.HasPrefix(core, "sh(") {
		return "Legacy P2SH script"
	}
	if strings.HasPrefix(core, "wsh(") {
		if strings.Contains(core, "multi(") {
			return "Native SegWit multisig wallet"
		}
		return "Native SegWit custom script"
	}
	if strings.HasPrefix(core, "tr(") {
		return "Taproot key path spend"
	}
	return "Unknown or unsupported descriptor type"
}

func main() {
	tests := []string{
		"wpkh([abcd1234/84h/0h/0h]xpub.../0/*)#3a1b2c3d",
		"sh(wpkh([abcd1234/49h/0h/0h]xpub.../0/*))#7f8e9d1c",
		"wsh(multi(2,[key1],[key2],[key3]))#abcd1234",
		"tr([f00dbabe/86h/0h/0h]xpub.../0/*)#c0ffee12",
	}

	for _, d := range tests {
		fmt.Printf("%s -> %s\n", d, interpretDescriptor(d))
	}
}
