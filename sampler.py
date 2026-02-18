import torch
import torch.nn as nn
from typing import Optional, Callable


class FlowMatchingSampler:
    """Base class for flow matching samplers."""
    
    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 50,
        device: str = "cpu",
        use_refiner: bool = False,
        prediction: str = "x"
    ):
        """
        Args:
            model: Trained DiT model that predicts velocity
            num_steps: Number of integration steps
            device: Device to run sampling on
            use_refiner: If True, use base + refiner velocity. If False, use only base.
        """
        self.model = model
        self.num_steps = num_steps
        self.device = device
        self.use_refiner = use_refiner
        self.prediction = prediction
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        y: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        use_refiner: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Generate samples from the flow.
        
        Args:
            batch_size: Number of samples to generate
            y: Optional class labels (batch_size,)
            noise: Optional pre-generated noise tensor. If None, will be sampled.
                  Must be shape (batch_size, channels, height, width)
            use_refiner: If provided, overrides the instance setting
        
        Returns:
            Generated samples of shape (batch_size, channels, height, width)
        """
        raise NotImplementedError("Subclasses must implement sample()")
    
    def _get_time_schedule(self) -> torch.Tensor:
        """Get linearly spaced timesteps from 0 to 1."""
        return torch.linspace(0, 1, self.num_steps, device=self.device)
    
    def _get_velocity(
        self,
        model_output,
        x_t: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        use_refiner: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Extract velocity from model output.
        
        Args:
            model_output: Either a single tensor (base velocity) or tuple of (base, refiner) velocities
            use_refiner: Whether to use base + refiner. If None, uses instance setting.
        
        Returns:
            Velocity tensor
        """
        use_refiner = use_refiner if use_refiner is not None else self.use_refiner
        
        # Check if output is a tuple (base and refiner)
        if isinstance(model_output, tuple):
            v_base, v_refiner = model_output
            
            if use_refiner:
                # Sum base and refiner velocities
                if self.prediction == "x":
                    return self.model._get_vt_from_x0(v_base+v_refiner, x_t, t)
                return v_base + v_refiner
            else:
                # Use only base velocity
                if self.prediction == "x":
                    return self.model._get_vt_from_x0(v_base, x_t, t)
                return v_base
        else:
            # Single output (legacy or base-only models)
            if self.prediction == "x":
                return self.model._get_vt_from_x0(model_output, x_t, t)
            return model_output


class EulerSampler(FlowMatchingSampler):
    """Simple Euler method for flow matching sampling.
    
    Uses first-order forward Euler integration:
    x_{t+dt} = x_t + v_t(x_t, t) * dt
    """
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        y: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        use_refiner: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Generate samples using Euler integration.
        
        Args:
            batch_size: Number of samples to generate
            y: Optional class labels (batch_size,)
            noise: Optional pre-generated noise. If None, will be sampled.
            use_refiner: If provided, overrides the instance setting
        
        Returns:
            Generated samples of shape (batch_size, channels, height, width)
        """
        use_refiner = use_refiner if use_refiner is not None else self.use_refiner
        
        # Initialize from noise: x ~ N(0, I)
        if noise is None:
            x = torch.randn(
                batch_size,
                self.model.in_channels,
                self.model.input_size,
                self.model.input_size,
                device=self.device
            )
        else:
            x = noise.to(self.device)
            batch_size = x.shape[0]
        
        # Get time schedule
        t_schedule = self._get_time_schedule()
        dt = 1.0 / self.num_steps
        
        # Move class labels to device if provided
        if y is not None:
            y = y.to(self.device)
        
        # Integrate forward in time
        for i in range(self.num_steps - 1):
            t = t_schedule[i]
            t_batch = t.expand(batch_size)
            
            # Predict velocity
            model_output = self.model(x, t_batch, y)
            v = self._get_velocity(model_output, x_t=x, t=t_batch, use_refiner=use_refiner)
            
            # Euler step: x = x + v * dt
            x = x + v * dt
        
        return x


class HeunSampler(FlowMatchingSampler):
    """Heun's method for flow matching sampling.
    
    Uses second-order Runge-Kutta (Heun) integration:
    k1 = v(x_t, t)
    k2 = v(x_t + dt*k1, t+dt)
    x_{t+dt} = x_t + (dt/2) * (k1 + k2)
    """
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        y: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        use_refiner: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Generate samples using Heun integration.
        
        Args:
            batch_size: Number of samples to generate
            y: Optional class labels (batch_size,)
            noise: Optional pre-generated noise. If None, will be sampled.
            use_refiner: If provided, overrides the instance setting
        
        Returns:
            Generated samples of shape (batch_size, channels, height, width)
        """
        use_refiner = use_refiner if use_refiner is not None else self.use_refiner
        
        # Initialize from noise: x ~ N(0, I)
        if noise is None:
            x = torch.randn(
                batch_size,
                self.model.in_channels,
                self.model.input_size,
                self.model.input_size,
                device=self.device
            )
        else:
            x = noise.to(self.device)
            batch_size = x.shape[0]
        
        # Get time schedule
        t_schedule = self._get_time_schedule()
        dt = 1.0 / self.num_steps
        
        # Move class labels to device if provided
        if y is not None:
            y = y.to(self.device)
        
        # Integrate forward in time using Heun's method
        for i in range(self.num_steps - 1):
            t_current = t_schedule[i]
            t_next = t_schedule[i + 1]
            
            t_current_batch = t_current.expand(batch_size)
            t_next_batch = t_next.expand(batch_size)
            
            # First stage: evaluate velocity at current point
            model_output = self.model(x, t_current_batch, y)
            v1 = self._get_velocity(model_output, x_t=x, t=t_current_batch, use_refiner=use_refiner)
            
            # Predictor: tentative step with first velocity
            x_pred = x + v1 * dt
            
            # Second stage: evaluate velocity at predicted point
            model_output = self.model(x_pred, t_next_batch, y)
            v2 = self._get_velocity(model_output, x_t=x_pred, t=t_next_batch, use_refiner=use_refiner)
            
            # Corrector: use average of velocities
            x = x + (v1 + v2) * (dt / 2.0)
        
        return x


class AdaptiveStepSampler(FlowMatchingSampler):
    """Adaptive step size sampler for flow matching.
    
    Uses error estimation to adaptively adjust step size during integration.
    Useful for efficient sampling when high accuracy is needed.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 50,
        device: str = "cpu",
        use_refiner: bool = False,
        atol: float = 1e-4,
        rtol: float = 1e-3,
        max_steps: int = 1000
    ):
        """
        Args:
            model: Trained DiT model
            num_steps: Initial number of steps
            device: Device to run on
            use_refiner: If True, use base + refiner velocity. If False, use only base.
            atol: Absolute tolerance for error control
            rtol: Relative tolerance for error control
            max_steps: Maximum number of steps allowed
        """
        super().__init__(model, num_steps, device, use_refiner=use_refiner)
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        y: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        use_refiner: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Generate samples using adaptive step size control.
        
        Args:
            batch_size: Number of samples to generate
            y: Optional class labels (batch_size,)
            noise: Optional pre-generated noise. If None, will be sampled.
            use_refiner: If provided, overrides the instance setting
        
        Returns:
            Generated samples of shape (batch_size, channels, height, width)
        """
        use_refiner = use_refiner if use_refiner is not None else self.use_refiner
        
        # Initialize from noise
        if noise is None:
            x = torch.randn(
                batch_size,
                self.model.in_channels,
                self.model.input_size,
                self.model.input_size,
                device=self.device
            )
        else:
            x = noise.to(self.device)
            batch_size = x.shape[0]
        
        if y is not None:
            y = y.to(self.device)
        
        t = torch.tensor(0.0, device=self.device)
        t_end = torch.tensor(1.0, device=self.device)
        
        # Adaptive integration
        dt = (t_end - t) / self.num_steps
        step_count = 0
        
        while t < t_end and step_count < self.max_steps:
            dt = min(dt, t_end - t)
            
            # Try Heun step
            t_batch = t.expand(batch_size)
            model_output = self.model(x, t_batch, y)
            v1 = self._get_velocity(model_output, use_refiner=use_refiner)
            
            x_heun = x + v1 * dt
            model_output = self.model(x_heun, (t + dt).expand(batch_size), y)
            v2 = self._get_velocity(model_output, use_refiner=use_refiner)
            x_heun = x + (v1 + v2) * (dt / 2.0)
            
            # Try Euler step (for error estimation)
            x_euler = x + v1 * dt
            
            # Estimate local error
            error = torch.abs(x_heun - x_euler).max()
            tolerance = self.atol + self.rtol * torch.abs(x).max()
            
            # Accept step if error is below tolerance
            if error <= tolerance or dt < 1e-6:
                x = x_heun
                t = t + dt
                
                # Increase step size if error is small
                if error < (tolerance / 10.0):
                    dt = dt * 1.5
            else:
                # Decrease step size and retry
                dt = dt * 0.5
            
            step_count += 1
        
        return x
