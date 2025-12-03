1. `graph_discrete_flow_model.py` —— `training_step`
	1.  将`batch_data`稀疏图转换为`dense_data`稠密图 
		- `utils.to_dense()` 详细过程在 [[从 0 开始的 DeFoG#^todense]]
	2. 对稠密图进行掩码`mask`
		```python
		def mask(self, node_mask, collapse=False):
			x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
			e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
			e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
			
        if collapse:
			self.X = torch.argmax(self.X, dim=-1)
			self.E = torch.argmax(self.E, dim=-1)
			self.X[node_mask == 0] = -1
			self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
		else:
			self.X = self.X * x_mask
			self.E = self.E * e_mask1 * e_mask2
			assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
		return self
		```